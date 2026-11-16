"""SimCluster: several simulated inference servers behind one front door.

``SimEngine`` is one server with the scheduler taken out. The problems one level up
are about *which* server gets a request, and how many servers there should be:
load balancing, cache-aware routing, autoscaling, draining, splitting prefill from
decode. For those the scheduler inside each server is not the point, so a
``SimCluster`` is a set of *replicas*, each a ``SimEngine`` with a fixed,
documented scheduler of its own, a prefix cache, and a life cycle.

    cluster = SimCluster(sim, CostModel(2**-6, 2**-12, 2**-9), replicas=3,
                         max_seqs=8, cache_tokens=4096, startup_s=2.0, route=my_route)
    workloads.drive(sim, requests, cluster.submit)     # each request goes through `route`
    sim.run()
    cluster.outcomes[request.id].first_token_s

Replicas
--------
A replica is ``starting`` for ``startup_s`` seconds after it is added (the initial
ones are ready at once), then ``ready``, then perhaps ``draining``, finally
``stopped``. What a router may read from one:

    replica.id                  0, 1, 2, ... in the order replicas were added; never reused
    replica.state               "starting" | "ready" | "draining" | "stopped"
    replica.queue_len           requests it holds: waiting plus running
    replica.waiting_len, replica.running_len
    replica.outstanding_tokens  prompt tokens still to prefill plus output tokens still
                                to produce, over all its requests
    replica.kv_usage            fraction of its KV blocks in use (0.0 without a pool)

``replica.cache`` is the replica's real prefix cache. It is there for tests: a router
that asks it has skipped its own work, which is to keep an index of what it sent
where.

The scheduler of a replica
--------------------------
Continuous batching, first come first served. Whenever its engine is free the
replica

1. admits waiting requests in arrival order while it has fewer than ``max_seqs``
   running (and, with a KV pool, while the pool can hold the request's whole context;
   a request that could never fit fails);
2. runs one step: every running sequence that is ready decodes; the rest of the
   step's ``max_batch_tokens`` is given to prefill, in admission order, as much per
   sequence as fits.

A step ends by the engine's rules (``STEP_DONE_PRIORITY``), so a request submitted at
the instant a step ends is in the next step. A replica with nothing to do waits.

Prefix cache
------------
With ``cache_tokens > 0``, a request that carries ``tokens`` (see
``workloads.shared_prefixes``) is matched against the prompts the replica has
prefilled: the longest common prefix with any of them, rounded down to a multiple of
``block_tokens`` and to at most ``prompt_tokens - 1``, is a *hit* and is not prefilled
again (``SimEngine.admit(..., cached_tokens=hit)``). A prompt enters the cache when
its prefill completes; the cache holds at most ``cache_tokens`` tokens and evicts the
least recently used prompts. Requests without ``tokens`` never hit.

Routing, scaling, draining
--------------------------
``cluster.submit(request)`` asks ``route(request, replicas)`` for a replica (or its
id) among the ready ones (the starting ones if none is ready yet) and hands the
request over; without a ``route`` the least ``queue_len`` wins, lowest id on ties.
``replica.submit(request)`` sends to one replica directly. Both return a future that
resolves to the request's ``Outcome`` and never raises.

``cluster.add_replica()`` starts one more. ``cluster.drain(replica)`` stops new work
from reaching it, lets what it holds finish, then stops it. ``cluster.stop(replica)``
stops it now: its unfinished requests **fail**. ``cluster.scale_to(n)`` adds replicas
or drains the newest ones until ``n`` are starting or ready. Sending to a draining or
stopped replica raises ``ClusterError`` (kept in ``cluster.violations``).

``cluster.metrics()`` is the snapshot an autoscaler reads; ``cluster.replica_seconds()``
is what the replicas have cost so far (time from being added to being stopped).

Prefill on one replica, decode on another
-----------------------------------------
``replica.submit(request, phase="prefill")`` runs the prompt only: the outcome arrives
with the first token, and the replica then drops the sequence. ``await
cluster.transfer(request, src, dst)`` takes ``transfer_s_per_token * prompt_tokens``
seconds, the KV cache moving. ``replica.submit(request, phase="decode")`` continues
from there (``SimEngine.admit(..., resume_from=1)``): no prefill, the remaining
``output_tokens - 1`` tokens. A request with one output token has nothing left to
decode.

Trace
-----
Every engine event carries ``engine=<replica id>`` (``SimEngine(name=...)``). The
cluster adds:

    route          id, replica                   a request was handed to a replica
    cache_lookup   id, replica, hit              at admission, for requests with tokens
    fail           id, replica, reason           "replica_stopped", "no_replica", "too_large"
    transfer       id, src, dst, tokens, duration     when a transfer starts
    transfer_done  id, src, dst
    replica_start, replica_ready, replica_drain, replica_stop      replica

Counters: ``routed``, ``finished``, ``failed``, ``cache_lookups``, ``cache_hit_tokens``,
``scale_ups``, ``scale_downs``, ``transfers``, ``transferred_tokens``.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from learnai.kit.counters import Counters
from learnai.kit.simengine import CostModel, SimEngine

PHASES = ("full", "prefill", "decode")
_COUNTERS = (
    "routed", "finished", "failed", "cache_lookups", "cache_hit_tokens", "scale_ups",
    "scale_downs", "transfers", "transferred_tokens",
)  # fmt: skip


class ClusterError(RuntimeError):
    """The cluster was asked for something it cannot do. ``code`` says what."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Outcome:
    """How a request ended on a replica.

    ``status`` is ``"finished"`` or ``"failed"`` (then ``reason`` says why and the
    times it never reached are None). ``first_token_s`` is None for a ``decode``
    phase, whose first token was produced elsewhere; ``finish_s`` of a ``prefill``
    phase is the time of its first token.
    """

    id: Hashable
    replica: int | None
    phase: str
    status: str
    submitted_s: float
    first_token_s: float | None = None
    finish_s: float | None = None
    cached_tokens: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class ClusterMetrics:
    """What an autoscaler sees. ``kv_usage`` is the largest among ready replicas."""

    now: float
    ready: int
    starting: int
    draining: int
    queue_depth: int  # requests waiting, over starting, ready, and draining replicas
    running: int
    outstanding_tokens: int
    kv_usage: float


class PrefixCache:
    """The prompts a replica has prefilled, least recently used first.

    ``match(tokens)`` is the number of leading tokens a new prompt can reuse: the
    longest common prefix with a stored prompt, in whole blocks, never the whole
    prompt. ``lookup`` is ``match`` that also marks the matched prompt as used.
    """

    def __init__(self, capacity_tokens: int, block_tokens: int) -> None:
        self.capacity_tokens = capacity_tokens
        self.block_tokens = block_tokens
        self._prompts: OrderedDict[tuple[int, ...], None] = OrderedDict()

    @property
    def cached_tokens(self) -> int:
        return sum(len(prompt) for prompt in self._prompts)

    def match(self, tokens: Sequence[int]) -> int:
        return self._best(tuple(tokens))[0]

    def lookup(self, tokens: Sequence[int]) -> int:
        hit, prompt = self._best(tuple(tokens))
        if prompt is not None:
            self._prompts.move_to_end(prompt)
        return hit

    def insert(self, tokens: Sequence[int]) -> None:
        prompt = tuple(tokens)
        if self.capacity_tokens <= 0 or len(prompt) > self.capacity_tokens:
            return
        self._prompts[prompt] = None
        self._prompts.move_to_end(prompt)
        while self.cached_tokens > self.capacity_tokens:
            self._prompts.popitem(last=False)

    def _best(self, tokens: tuple[int, ...]) -> tuple[int, tuple[int, ...] | None]:
        best, which = 0, None
        for prompt in self._prompts:
            shared = 0
            for a, b in zip(tokens, prompt, strict=False):
                if a != b:
                    break
                shared += 1
            shared = min(shared, len(tokens) - 1) // self.block_tokens * self.block_tokens
            if shared > best:
                best, which = shared, prompt
        return best, which


@dataclass
class _Job:
    request: Any
    phase: str
    future: asyncio.Future
    submitted_s: float
    cached_tokens: int = 0
    first_token_s: float | None = None
    worst_blocks: int = 0


@dataclass(frozen=True)
class _PrefillOnly:
    """What the engine is given for a ``prefill`` phase: the prompt, and one token
    more than will be taken, so the sequence is still live when it is dropped."""

    id: Hashable
    prompt_tokens: int
    output_tokens: int = 2


class Replica:
    """One simulated server of a ``SimCluster``. See the module docstring."""

    def __init__(self, cluster: SimCluster, replica_id: int, startup_s: float) -> None:
        self.cluster = cluster
        self.id = replica_id
        self.state = "starting" if startup_s > 0 else "ready"
        self.started_s = cluster.sim.now()
        self.stopped_s: float | None = None
        self.engine = SimEngine(
            cluster.sim, cluster.cost, max_batch_tokens=cluster.max_batch_tokens,
            max_seqs=cluster.max_seqs, kv_blocks=cluster.kv_blocks,
            block_tokens=cluster.block_tokens, name=replica_id,
        )  # fmt: skip
        self.cache = PrefixCache(cluster.cache_tokens, cluster.block_tokens)
        self._waiting: deque[_Job] = deque()
        self._running: dict[Hashable, _Job] = {}
        self._wake = asyncio.Event()
        cluster.sim.trace.record("replica_start", replica=replica_id)
        if self.state == "ready":
            cluster.sim.trace.record("replica_ready", replica=replica_id)
        cluster.sim.spawn(self._run(startup_s))

    # -- what a router reads ----------------------------------------------------------------

    @property
    def waiting_len(self) -> int:
        return len(self._waiting)

    @property
    def running_len(self) -> int:
        return len(self._running)

    @property
    def queue_len(self) -> int:
        return len(self._waiting) + len(self._running)

    @property
    def outstanding_tokens(self) -> int:
        total = 0
        for job in self._waiting:
            total += _work(job)
        live = set(self.engine.live)
        for seq_id, job in self._running.items():
            if seq_id not in live:
                continue  # its last token came in a step that ends at this very instant
            done = self.engine.generated(seq_id)
            goal = 1 if job.phase == "prefill" else job.request.output_tokens
            total += self.engine.remaining_prefill(seq_id) + max(0, goal - done)
        return total

    @property
    def kv_usage(self) -> float:
        if self.engine.kv_blocks is None:
            return 0.0
        return (self.engine.kv_blocks - self.engine.free_blocks) / self.engine.kv_blocks

    # -- taking work --------------------------------------------------------------------------

    def submit(self, request: Any, *, phase: str = "full") -> asyncio.Future:
        """Give ``request`` to this replica. Returns a future of its ``Outcome``.

        ``phase`` is ``"full"`` (prefill and decode here), ``"prefill"``, or
        ``"decode"`` (module docstring). Raises ``ClusterError`` for a replica that is
        draining or stopped (``replica_unavailable``) and for a ``decode`` phase with
        nothing to decode; ``ValueError`` for an unknown phase.
        """
        cluster = self.cluster
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {', '.join(PHASES)}, got {phase!r}")
        if self.state in ("draining", "stopped"):
            raise cluster._violation(
                "replica_unavailable",
                f"request {request.id!r} was sent to replica {self.id}, which is {self.state}",
            )
        if phase == "decode" and request.output_tokens < 2:
            raise cluster._violation(
                "nothing_to_decode",
                f"request {request.id!r} has one output token, produced by its prefill",
            )
        job = _Job(request, phase, cluster.sim.loop.create_future(), cluster.sim.now())
        context = request.prompt_tokens + request.output_tokens - 1
        job.worst_blocks = -(-context // cluster.block_tokens)
        cluster.counters.add("routed")
        cluster.sim.trace.record("route", id=request.id, replica=self.id)
        if cluster.kv_blocks is not None and job.worst_blocks > cluster.kv_blocks:
            self._fail(job, "too_large")
            return job.future
        self._waiting.append(job)
        self._wake.set()
        return job.future

    # -- the server ---------------------------------------------------------------------------

    async def _run(self, startup_s: float) -> None:
        sim = self.cluster.sim
        if startup_s > 0:
            await sim.sleep(startup_s)
            if self.state == "starting":
                self.state = "ready"
                sim.trace.record("replica_ready", replica=self.id)
        while self.state != "stopped":
            self._admit()
            prefill, decode = self._plan()
            if not prefill and not decode:
                if self.state == "draining" and not self._waiting and not self._running:
                    self._stop()
                    return
                self._wake.clear()
                await self._wake.wait()
                continue
            result = await self.engine.step(prefill=prefill, decode=decode)
            if self.state == "stopped":
                return
            self._after_step(result)

    def _admit(self) -> None:
        cluster, engine = self.cluster, self.engine
        while self._waiting and (cluster.max_seqs is None or len(self._running) < cluster.max_seqs):
            job = self._waiting[0]
            if cluster.kv_blocks is not None:
                reserved = sum(running.worst_blocks for running in self._running.values())
                if reserved + job.worst_blocks > cluster.kv_blocks:
                    break  # first come, first served: nobody overtakes the head
            self._waiting.popleft()
            request = job.request
            if job.phase == "decode":
                engine.admit(request, resume_from=1)
            else:
                tokens = getattr(request, "tokens", None)
                if tokens is not None and cluster.cache_tokens > 0:
                    job.cached_tokens = self.cache.lookup(tokens)
                    cluster.counters.add("cache_lookups")
                    cluster.counters.add("cache_hit_tokens", job.cached_tokens)
                    cluster.sim.trace.record(
                        "cache_lookup", id=request.id, replica=self.id, hit=job.cached_tokens
                    )
                target = request
                if job.phase == "prefill":
                    target = _PrefillOnly(request.id, request.prompt_tokens)
                engine.admit(target, cached_tokens=job.cached_tokens)
            self._running[request.id] = job

    def _plan(self) -> tuple[list[tuple[Hashable, int]], list[Hashable]]:
        engine = self.engine
        budget = engine.max_batch_tokens
        decode = [seq_id for seq_id in self._running if engine.remaining_prefill(seq_id) == 0]
        if budget is not None:
            decode = decode[:budget]
        left = None if budget is None else budget - len(decode)
        prefill = []
        for seq_id in self._running:
            remaining = engine.remaining_prefill(seq_id)
            if remaining == 0:
                continue
            chunk = remaining if left is None else min(remaining, left)
            if chunk < 1:
                break
            prefill.append((seq_id, chunk))
            if left is not None:
                left -= chunk
        return prefill, decode

    def _after_step(self, result: Any) -> None:
        sim = self.cluster.sim
        for seq_id in result.tokens:
            job = self._running.get(seq_id)
            if job is None or job.phase == "decode" or job.first_token_s is not None:
                continue
            job.first_token_s = sim.now()
            tokens = getattr(job.request, "tokens", None)
            if tokens is not None:
                self.cache.insert(tokens)  # the prompt's keys and values now exist here
            if job.phase == "prefill":
                self.engine.release(seq_id)  # the KV cache leaves with the request
                self._finish(job)
        for seq_id in result.finished:
            job = self._running.get(seq_id)
            if job is not None:
                self._finish(job)

    def _finish(self, job: _Job) -> None:
        cluster = self.cluster
        del self._running[job.request.id]
        outcome = Outcome(
            job.request.id, self.id, job.phase, "finished", job.submitted_s,
            job.first_token_s, cluster.sim.now(), job.cached_tokens,
        )  # fmt: skip
        cluster.counters.add("finished")
        cluster._settle(job, outcome)

    def _fail(self, job: _Job, reason: str) -> None:
        cluster = self.cluster
        outcome = Outcome(
            job.request.id, self.id, job.phase, "failed", job.submitted_s,
            job.first_token_s, None, job.cached_tokens, reason,
        )  # fmt: skip
        cluster.counters.add("failed")
        cluster.sim.trace.record("fail", id=job.request.id, replica=self.id, reason=reason)
        cluster._settle(job, outcome)

    def _stop(self) -> None:
        """Stop now: whatever is unfinished fails."""
        if self.state == "stopped":
            return
        self.state = "stopped"
        self.stopped_s = self.cluster.sim.now()
        for seq_id, job in list(self._running.items()):
            self.engine.release(seq_id)
            del self._running[seq_id]
            self._fail(job, "replica_stopped")
        while self._waiting:
            self._fail(self._waiting.popleft(), "replica_stopped")
        self.cluster.sim.trace.record("replica_stop", replica=self.id)
        self._wake.set()

    def __repr__(self) -> str:
        return (
            f"Replica({self.id}, {self.state}, waiting={self.waiting_len}, "
            f"running={self.running_len})"
        )


def _work(job: _Job) -> int:
    request = job.request
    if job.phase == "prefill":
        return request.prompt_tokens + 1
    if job.phase == "decode":
        return request.output_tokens - 1
    return request.prompt_tokens + request.output_tokens


class SimCluster:
    """Replicas of ``SimEngine`` behind a ``route`` hook. See the module docstring.

    Args:
        sim: the simulated clock (``ctx.simclock()``).
        cost: the ``CostModel`` of every replica's engine.
        replicas: how many replicas exist, ready, at the start.
        max_seqs: sequences a replica runs at once; None for no limit.
        max_batch_tokens, kv_blocks, block_tokens: passed to each ``SimEngine``.
        cache_tokens: capacity of each replica's prefix cache, in tokens; 0: no cache.
        startup_s: how long a replica added later is ``starting`` before it is ready.
        route: ``route(request, replicas) -> replica or replica id``; None for least
            ``queue_len``.
        transfer_s_per_token: the cost of ``transfer``, per prompt token.

    Attributes:
        replicas: the replicas that are not stopped, in id order.
        all_replicas: every replica ever added.
        outcomes: ``{request id: Outcome}`` of requests that finished or failed. For a
            request served in two phases, the outcome of the phase that ended last.
        violations: every ``ClusterError`` raised.
        counters: see the module docstring.
    """

    def __init__(
        self,
        sim: Any,
        cost: CostModel,
        *,
        replicas: int = 1,
        max_seqs: int | None = 8,
        max_batch_tokens: int | None = None,
        kv_blocks: int | None = None,
        block_tokens: int = 16,
        cache_tokens: int = 0,
        startup_s: float = 0.0,
        route: Callable[[Any, list[Replica]], Any] | None = None,
        transfer_s_per_token: float = 0.0,
    ) -> None:
        if replicas < 0 or startup_s < 0 or cache_tokens < 0 or transfer_s_per_token < 0:
            raise ValueError("replicas, startup_s, cache_tokens, and transfer cost must be >= 0")
        if max_seqs is not None and max_seqs < 1:
            raise ValueError("max_seqs must be at least 1")
        if max_batch_tokens is not None and max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be at least 1")
        self.sim = sim
        self.cost = cost
        self.max_seqs = max_seqs
        self.max_batch_tokens = max_batch_tokens
        self.kv_blocks = kv_blocks
        self.block_tokens = block_tokens
        self.cache_tokens = cache_tokens
        self.startup_s = startup_s
        self.route = route
        self.transfer_s_per_token = transfer_s_per_token
        self.counters = Counters(*_COUNTERS)
        self.all_replicas: list[Replica] = []
        self.outcomes: dict[Hashable, Outcome] = {}
        self.violations: list[ClusterError] = []
        for _ in range(replicas):
            self.all_replicas.append(Replica(self, len(self.all_replicas), 0.0))

    # -- replicas ----------------------------------------------------------------------------

    @property
    def replicas(self) -> list[Replica]:
        return [replica for replica in self.all_replicas if replica.state != "stopped"]

    def replica(self, replica: Replica | int) -> Replica:
        """The replica with this id (or the replica itself)."""
        if isinstance(replica, Replica):
            return replica
        if isinstance(replica, bool) or not isinstance(replica, int):
            raise TypeError(f"a replica is a Replica or its int id, got {replica!r}")
        if not 0 <= replica < len(self.all_replicas):
            raise ValueError(f"there is no replica {replica}")
        return self.all_replicas[replica]

    def add_replica(self) -> Replica:
        """Start one more replica; it is ready ``startup_s`` seconds from now."""
        new = Replica(self, len(self.all_replicas), self.startup_s)
        self.all_replicas.append(new)
        self.counters.add("scale_ups")
        return new

    def drain(self, replica: Replica | int) -> None:
        """No new work for this replica; it stops when what it holds has finished."""
        target = self.replica(replica)
        if target.state in ("draining", "stopped"):
            return
        target.state = "draining"
        self.counters.add("scale_downs")
        self.sim.trace.record("replica_drain", replica=target.id)
        target._wake.set()

    def stop(self, replica: Replica | int) -> None:
        """Stop this replica now. Its unfinished requests fail."""
        target = self.replica(replica)
        if target.state == "stopped":
            return
        if target.state != "draining":
            self.counters.add("scale_downs")
        target._stop()

    def scale_to(self, n: int) -> None:
        """Make ``n`` the number of replicas that are starting or ready.

        Too few: replicas are added. Too many: the newest are drained (so nothing
        they hold fails).
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"scale_to needs an int >= 0, got {n!r}")
        serving = [r for r in self.all_replicas if r.state in ("starting", "ready")]
        for _ in range(n - len(serving)):
            self.add_replica()
        for replica in reversed(serving[n:]):
            self.drain(replica)

    def metrics(self) -> ClusterMetrics:
        """The numbers an autoscaler decides by, now."""
        live = self.replicas
        ready = [r for r in live if r.state == "ready"]
        return ClusterMetrics(
            now=self.sim.now(),
            ready=len(ready),
            starting=sum(1 for r in live if r.state == "starting"),
            draining=sum(1 for r in live if r.state == "draining"),
            queue_depth=sum(r.waiting_len for r in live),
            running=sum(r.running_len for r in live),
            outstanding_tokens=sum(r.outstanding_tokens for r in live),
            kv_usage=max((r.kv_usage for r in ready), default=0.0),
        )

    def replica_seconds(self) -> float:
        """Seconds of replica time paid for so far: from being added to being stopped."""
        now = self.sim.now()
        return sum(
            (now if r.stopped_s is None else r.stopped_s) - r.started_s for r in self.all_replicas
        )

    # -- requests ----------------------------------------------------------------------------

    def submit(self, request: Any) -> asyncio.Future:
        """Route ``request`` and hand it to the chosen replica; a future of its ``Outcome``.

        With no replica starting or ready, the request fails at once (``no_replica``).
        """
        candidates = [r for r in self.all_replicas if r.state == "ready"]
        if not candidates:
            candidates = [r for r in self.all_replicas if r.state == "starting"]
        if not candidates:
            job = _Job(request, "full", self.sim.loop.create_future(), self.sim.now())
            outcome = Outcome(request.id, None, "full", "failed", job.submitted_s,
                              reason="no_replica")  # fmt: skip
            self.counters.add("failed")
            self.sim.trace.record("fail", id=request.id, replica=None, reason="no_replica")
            self._settle(job, outcome)
            return job.future
        if self.route is None:
            chosen: Any = min(candidates, key=lambda r: (r.queue_len, r.id))
        else:
            chosen = self.route(request, list(candidates))
        return self.replica(chosen).submit(request)

    async def transfer(self, request: Any, src: Replica | int, dst: Replica | int) -> None:
        """Move ``request``'s KV cache from ``src`` to ``dst``: takes
        ``transfer_s_per_token * request.prompt_tokens`` seconds."""
        a, b = self.replica(src), self.replica(dst)
        duration = self.transfer_s_per_token * request.prompt_tokens
        self.counters.add("transfers")
        self.counters.add("transferred_tokens", request.prompt_tokens)
        self.sim.trace.record(
            "transfer", id=request.id, src=a.id, dst=b.id, tokens=request.prompt_tokens,
            duration=duration,
        )  # fmt: skip
        await self.sim.sleep(duration)
        self.sim.trace.record("transfer_done", id=request.id, src=a.id, dst=b.id)

    # -- internals ---------------------------------------------------------------------------

    def _settle(self, job: _Job, outcome: Outcome) -> None:
        self.outcomes[job.request.id] = outcome
        if not job.future.done():
            job.future.set_result(outcome)

    def _violation(self, code: str, message: str) -> ClusterError:
        error = ClusterError(code, message)
        self.violations.append(error)
        self.sim.trace.record("violation", code=code, message=message)
        return error

    def __repr__(self) -> str:
        states = ", ".join(f"{r.id}:{r.state}" for r in self.all_replicas)
        return f"SimCluster({states}; {self.counters!r})"
