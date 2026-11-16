# BUG: there is no counter lift: a tenant that was idle returns with its old, low counter and is served ahead of everyone until it has caught up.
import asyncio
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Rejected:
    """What `submit` answers for a request the gateway did not accept."""

    reason: str  # "rate_limited" or "queue_full"
    retry_after_s: float | None = None
    status: str = "rejected"


def common_prefix(a, b) -> int:
    """The number of leading tokens two prompts share."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


class Gateway:
    def __init__(self, cluster, cfg):
        self.cluster = cluster
        self.cfg = cfg
        self.counters = {}  # stage 4: tenant -> virtual token counter
        self._queue = []  # admitted requests that are not on a replica yet, oldest first
        self._queued_tokens = 0
        self._clients = {}  # request id -> the future `submit` returned
        self._buckets = {}  # stage 1: tenant -> (balance, time of the balance)
        self._index = {}  # stage 3: replica id -> the last prompts routed to it
        self._last = None  # stage 4: tenant of the request dispatched last
        self._history = []  # stage 5: (time, recommendation) of recent evaluations

    # -- stage 1: admission -------------------------------------------------------------

    def submit(self, request) -> asyncio.Future:
        cfg, now = self.cfg, self.cluster.sim.now()
        future = self.cluster.sim.loop.create_future()
        cost = request.prompt_tokens

        # Decide first, charge second: a rejected request must leave no trace.
        balance = None
        if cfg.rate_tokens_per_s is not None:
            balance = self._balance(request.tenant, now)
            if balance < cost:
                retry = None
                if cost <= cfg.burst_tokens:
                    retry = (cost - balance) / cfg.rate_tokens_per_s
                future.set_result(Rejected("rate_limited", retry))
                return future
        if (
            len(self._queue) + 1 > cfg.max_queued_requests
            or self._queued_tokens + cost > cfg.max_queued_tokens
        ):
            future.set_result(Rejected("queue_full"))
            return future
        if balance is not None:
            self._buckets[request.tenant] = (balance - cost, now)

        if cfg.fair:
            self._lift(request.tenant)
        self._queue.append(request)
        self._queued_tokens += cost
        self._clients[request.id] = future
        self._dispatch()
        return future

    def _balance(self, tenant, now) -> float:
        cfg = self.cfg
        balance, since = self._buckets.get(tenant, (cfg.burst_tokens, now))
        return min(cfg.burst_tokens, balance + cfg.rate_tokens_per_s * (now - since))

    # -- the gateway queue ----------------------------------------------------------------

    def _dispatch(self) -> None:
        """Send queued requests to replicas for as long as a replica has a free slot."""
        while self._queue:
            free = [
                replica
                for replica in self.cluster.replicas
                if replica.state == "ready" and replica.queue_len < self.cfg.replica_slots
            ]
            if not free:
                return
            request = self._next_request()
            self._queue.remove(request)
            self._queued_tokens -= request.prompt_tokens
            if self.cfg.fair:
                # Input tokens are charged when the request leaves the queue.
                self.counters[request.tenant] += self.cfg.w_in * request.prompt_tokens
                self._last = request.tenant
            replica = self._choose(request, free)
            outcome = replica.submit(request)
            outcome.add_done_callback(lambda done, request=request: self._finished(request, done))

    def _finished(self, request, done) -> None:
        outcome = done.result()
        if self.cfg.fair and outcome.status == "finished":
            # The gateway sees an answer when it is complete: output tokens are charged now.
            self.counters[request.tenant] += self.cfg.w_out * request.output_tokens
        self._clients.pop(request.id).set_result(outcome)
        self._dispatch()  # a slot is free

    # -- stages 2 and 3: which replica ------------------------------------------------------

    def _choose(self, request, free):
        def least_loaded():
            return min(free, key=lambda r: (r.outstanding_tokens, r.id))

        cfg, tokens = self.cfg, request.tokens
        if not cfg.prefix_routing or tokens is None:
            return least_loaded()
        least = min(replica.outstanding_tokens for replica in free)
        # Affinity may only choose among replicas that are not much busier than the
        # idlest one: this is what stops a hot prefix from burying one replica.
        eligible = [r for r in free if r.outstanding_tokens - least <= cfg.imbalance_tokens]
        best = max(
            eligible, key=lambda r: (self._match(tokens, r.id), -r.outstanding_tokens, -r.id)
        )
        if self._match(tokens, best.id) >= cfg.match_threshold * len(tokens):
            chosen = best
        else:
            chosen = least_loaded()
        self._index.setdefault(chosen.id, deque(maxlen=cfg.index_size)).append(tokens)
        return chosen

    def _match(self, tokens, replica_id) -> int:
        prompts = self._index.get(replica_id, ())
        return max((common_prefix(tokens, prompt) for prompt in prompts), default=0)

    # -- stage 4: which request ---------------------------------------------------------------

    def _next_request(self):
        if not self.cfg.fair:
            return self._queue[0]
        # The earliest request of the queued tenant with the smallest counter. `min`
        # keeps the first of equal keys, so ties go to the request first in the queue.
        return min(self._queue, key=lambda r: self.counters[r.tenant])

    def _lift(self, tenant) -> None:
        counter = self.counters.setdefault(tenant, 0.0)
        self.counters[tenant] = counter

    # -- stage 5: how many replicas -------------------------------------------------------------

    def evaluate(self) -> int:
        cfg, cluster = self.cfg, self.cluster
        metrics = cluster.metrics()
        current = metrics.ready + metrics.starting  # replicas that are starting count
        # The load is the requests in the house: here most of the waiting ones are in
        # the gateway's own queue, which the cluster cannot see.
        load = len(self._queue) + metrics.queue_depth + metrics.running
        capacity = current * cfg.target
        if abs(load - capacity) <= cfg.tolerance * capacity:
            raw = current  # close enough: do not chase noise
        else:
            raw = -(-load // cfg.target)
        raw = max(cfg.min_replicas, min(cfg.max_replicas, raw))
        # Remember this recommendation; forget those older than the window.
        self._history = [(t, n) for t, n in self._history if metrics.now - t < cfg.window_s]
        self._history.append((metrics.now, raw))
        if raw >= current:
            n = raw  # scale up at once
        else:
            n = min(current, max(n for _, n in self._history))  # scale down late
        # scale_to drains the newest replicas: they finish what they hold.
        cluster.scale_to(n)
        self._dispatch()  # a replica that has become ready may take queued work
        return n
