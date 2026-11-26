"""Tests for pipeline-parallel-schedule.

`Pipeline` (below) is the simulated hardware: stages with service times, links with
transfer times, on a simulated clock whose unit is one millisecond. It validates
every call against the physical rules of the statement by *time* (a stage is free
from the instant its job ends, whatever order the event loop delivers callbacks in)
and records every start. A case runs the solution, checks the rules and the returned
tokens, recomputes the validity of the schedule from the recorded starts, and compares
the starts with the oracle's, which the statement's scheduling rule determines
uniquely. Makespan and throughput are reported against two baselines computed here:
one token in the pipeline at a time, and the bottleneck stage's rate.

The kit has no pipeline object (one problem uses it), so it lives in this file.
"""

import asyncio
import copy

from learnai.judge import case, check


class PipelineError(RuntimeError):
    """A call that the hardware cannot carry out at this moment."""


class Pipeline:
    def __init__(self, sim, service_ms, transfer_ms, requests):
        check.that(len(transfer_ms) == len(service_ms) - 1, "test bug: one link per stage pair")
        self.sim = sim
        self.service_ms = tuple(service_ms)
        self.transfer_ms = tuple(transfer_ms)
        self.n_stages = len(self.service_ms)
        self._requests = {
            seq: (index, arrival, n) for index, (seq, arrival, n) in enumerate(requests)
        }
        self._free_at = [0] * self.n_stages
        self._done_at = {}  # (stage, seq, token) -> when that run ends
        self._arrives_at = {}  # (stage, seq, token) -> when the activation reaches `stage`
        self.starts = []  # (time, stage, seq, token)
        self.transfers = []  # (time, stage, seq, token)
        self.violations = []

    def _fail(self, message):
        self.violations.append(message)
        raise PipelineError(message)

    def _job(self, what, stage, seq, token, last_stage):
        if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= last_stage:
            self._fail(f"{what}: there is no stage {stage!r}")
        if seq not in self._requests:
            self._fail(f"{what}: there is no sequence {seq!r}")
        n_tokens = self._requests[seq][2]
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < n_tokens:
            self._fail(f"{what}: sequence {seq!r} has no token {token!r}")

    async def run(self, stage, seq, token):
        """Stage `stage` computes token `token` of sequence `seq`."""
        now = self.sim.now()
        what = f"run({stage}, {seq!r}, {token}) at t={now:g} ms"
        self._job(what, stage, seq, token, self.n_stages - 1)
        key = (stage, seq, token)
        if key in self._done_at:
            self._fail(f"{what}: this stage has already run this token")
        if now < self._free_at[stage]:
            self._fail(f"{what}: stage {stage} is busy until t={self._free_at[stage]:g} ms")
        if stage > 0:
            arrives = self._arrives_at.get(key)
            if arrives is None:
                self._fail(f"{what}: the activation was never transferred from stage {stage - 1}")
            if now < arrives:
                self._fail(f"{what}: the activation is in transit until t={arrives:g} ms")
        elif token == 0:
            arrival = self._requests[seq][1]
            if now < arrival:
                self._fail(f"{what}: the request arrives at t={arrival:g} ms")
        else:
            previous = self._done_at.get((self.n_stages - 1, seq, token - 1))
            if previous is None or now < previous:
                when = (
                    "has not entered the last stage" if previous is None else f"t={previous:g} ms"
                )
                self._fail(
                    f"{what}: token {token - 1} of this sequence has not left the last stage "
                    f"({when}), and the model needs it as input"
                )
        done = now + self.service_ms[stage]
        self._free_at[stage] = done
        self._done_at[key] = done
        self.starts.append((now, stage, seq, token))
        await asyncio.sleep(self.service_ms[stage])
        if stage == self.n_stages - 1:
            return token_id(self._requests[seq][0], token)
        return None

    async def transfer(self, stage, seq, token):
        """Move the output of `stage` for this token to stage `stage + 1`."""
        now = self.sim.now()
        what = f"transfer({stage}, {seq!r}, {token}) at t={now:g} ms"
        self._job(what, stage, seq, token, self.n_stages - 2)
        done = self._done_at.get((stage, seq, token))
        if done is None or now < done:
            self._fail(f"{what}: stage {stage} has not finished this token")
        key = (stage + 1, seq, token)
        if key in self._arrives_at:
            self._fail(f"{what}: this activation was already transferred")
        self._arrives_at[key] = now + self.transfer_ms[stage]
        self.transfers.append((now, stage, seq, token))
        await asyncio.sleep(self.transfer_ms[stage])


def token_id(index, token):
    return (index * 7919 + token * 104729 + 17) % 50000


def timeline(pipe, requests, starts, limit=60):
    lines = [
        f"service_ms={list(pipe.service_ms)}, transfer_ms={list(pipe.transfer_ms)}",
        "requests (id, arrival_ms, n_tokens): "
        + ", ".join(str(tuple(r)) for r in requests[:12])
        + (" ..." if len(requests) > 12 else ""),
        "starts (time: stage <- sequence[token]):",
    ]
    for t, stage, seq, token in sorted(starts)[:limit]:
        lines.append(f"  t={t:>5g}: stage {stage} <- {seq}[{token}]")
    if len(starts) > limit:
        lines.append(f"  ... {len(starts) - limit} more")
    return "\n".join(lines) + "\n"


def execute(ctx, module, service_ms, transfer_ms, requests):
    """Run `module.run_pipeline`; returns (pipe, outputs, time it returned)."""
    sim = ctx.simclock()
    pipe = Pipeline(sim, service_ms, transfer_ms, requests)
    try:
        outputs = sim.run_until_complete(module.run_pipeline(pipe, [tuple(r) for r in requests]))
    except PipelineError as exc:
        check.that(
            False,
            f"the pipeline refused a call: {exc}",
            timeline(pipe, requests, pipe.starts),
        )
    returned = sim.now()
    outputs = copy.deepcopy(outputs)  # what was returned, before anything still running adds to it
    sim.run()  # let anything still in flight finish, so that the record is complete
    return pipe, outputs, returned


def sequential_ms(service_ms, transfer_ms, requests):
    """Makespan with one token in the pipeline at a time, requests in order."""
    per_token = sum(service_ms) + sum(transfer_ms)
    now = 0
    for _, arrival, n_tokens in requests:
        now = max(now, arrival) + n_tokens * per_token
    return now


def play(ctx, service_ms, transfer_ms, requests, *, literal=None):
    pipe, outputs, returned = execute(ctx, ctx.solution, service_ms, transfer_ms, requests)
    shown = timeline(pipe, requests, pipe.starts)
    if pipe.violations:  # raised inside the solution and swallowed there
        check.that(False, f"the pipeline refused a call: {pipe.violations[0]}", shown)

    # Everything was computed, and the tokens came back in order.
    last = pipe.n_stages - 1
    missing = [
        (seq, token)
        for seq, _, n_tokens in requests
        for token in range(n_tokens)
        if (last, seq, token) not in pipe._done_at
    ]
    check.that(
        not missing,
        f"{len(missing)} tokens never went through the last stage",
        f"first missing: {missing[:5]}\n{shown}",
    )
    finish = max(pipe._done_at.values())
    check.that(
        returned >= finish,
        f"run_pipeline returned at t={returned:g} ms, before the last token was finished",
        f"the last stage finished at t={finish:g} ms\n{shown}",
    )
    want_outputs = {
        seq: [token_id(index, t) for t in range(n)] for index, (seq, _, n) in enumerate(requests)
    }
    check.that(isinstance(outputs, dict), "run_pipeline: must return a dict", f"got: {outputs!r}")
    check.equal(sorted(outputs), sorted(want_outputs), label="sequence ids of the result")
    for seq, tokens in want_outputs.items():
        check.that(
            list(outputs[seq]) == tokens,
            f"the tokens returned for sequence {seq!r} are not the last stage's results in "
            "token order",
            f"expected {tokens}\ngot {outputs[seq]!r}\n",
        )

    # The schedule is the one the rule determines.
    oracle, _, want_returned = execute(ctx, ctx.oracle, service_ms, transfer_ms, requests)
    got, want = sorted(pipe.starts), sorted(oracle.starts)
    if literal is not None:
        check.that(want == sorted(literal), "test bug: literal schedule and oracle disagree",
                   timeline(oracle, requests, oracle.starts))  # fmt: skip
    detail = f"your schedule:\n{shown}\nthe schedule the rule gives:\n" + timeline(
        oracle, requests, oracle.starts
    )
    for g, w in zip(got, want, strict=False):
        if g == w:
            continue
        if (w[0], w[1]) < (g[0], g[1]):
            message = (
                f"stage {w[1]} started nothing at t={w[0]:g} ms although it was free and a "
                f"job was ready (expected token {w[3]} of {w[2]!r})"
            )
        else:
            message = (
                f"at t={g[0]:g} ms stage {g[1]} started token {g[3]} of {g[2]!r}; the rule's "
                f"next start is token {w[3]} of {w[2]!r} on stage {w[1]} at t={w[0]:g} ms"
            )
        check.that(False, message, detail)
    check.that(len(got) == len(want), f"the pipeline ran {len(got)} jobs, expected "
               f"{len(want)} (one per stage and token)", detail)  # fmt: skip
    late = [
        (t, stage, seq, token)
        for t, stage, seq, token in pipe.transfers
        if t != pipe._done_at[(stage, seq, token)]
    ]
    check.that(
        not late,
        "a transfer did not start at the instant its stage finished",
        f"first: transfer{late[0][1:] if late else ''} started at t={late[0][0] if late else 0:g}"
        f" ms\n{shown}",
    )

    n_tokens = sum(n for _, _, n in requests)
    ctx.metric("one_token_at_a_time", sequential_ms(service_ms, transfer_ms, requests), unit="ms")
    ctx.metric("tokens_per_s", 1000 * n_tokens / returned, unit="tokens/s", higher_is_better=True)
    check.at_most("makespan", returned, want_returned, unit="ms")
    return pipe, returned


EXAMPLE = [("a", 0, 2), ("b", 0, 2), ("c", 4, 1)]
EXAMPLE_STARTS = [
    (0, 0, "a", 0), (2, 0, "b", 0), (3, 1, "a", 0), (4, 0, "c", 0), (6, 0, "a", 1),
    (6, 1, "b", 0), (9, 0, "b", 1), (9, 1, "c", 0), (12, 1, "a", 1), (15, 1, "b", 1),
]  # fmt: skip


# Rejects one_sequence_at_a_time and one_token_in_flight (30 ms), free_transfers,
# tokens_overlap, priority_by_request, transfer_inside_next_stage.
@case("the worked example: two stages, three sequences, done at 18 ms instead of 30",
      public=True)  # fmt: skip
def worked_example(ctx):
    _, returned = play(ctx, (2, 3), (1,), EXAMPLE, literal=EXAMPLE_STARTS)
    check.equal(returned, 18, label="time at which run_pipeline returned (ms)")


# Rejects greedy_locks and ties_to_newcomer.
@case("a tie at the first stage: a returning token and a new request at t=5", public=True)
def tie_at_first_stage(ctx):
    literal = [(0, 0, "a", 0), (3, 1, "a", 0), (5, 0, "a", 1), (7, 0, "b", 0), (8, 1, "a", 1),
               (10, 1, "b", 0)]  # fmt: skip
    play(ctx, (2, 2), (1,), [("a", 0, 2), ("b", 5, 1)], literal=literal)
    # The same tie while the stage is still busy: both wait, the earlier request goes first.
    play(ctx, (4, 2), (1,), [("a", 0, 3), ("b", 1, 2), ("c", 7, 2), ("d", 7, 1)])


# Rejects tokens_overlap and free_transfers.
@case("one sequence alone: its tokens go through one after the other", public=True)
def single_sequence(ctx):
    _, returned = play(ctx, (3, 1, 2), (1, 2), [("a", 1, 3)])
    check.equal(returned, 28, label="time at which run_pipeline returned (ms)")
    play(ctx, (5,), (), [("only", 3, 4)])


def saturated(n_sequences, n_tokens):
    return [(f"s{i}", 0, n_tokens) for i in range(n_sequences)]


# Rejects one_sequence_at_a_time, one_token_in_flight, transfer_inside_next_stage.
@case("eight sequences of 30 tokens on four stages: throughput within 5 percent of the "
      "bottleneck stage's rate", public=True)  # fmt: skip
def bottleneck_throughput(ctx):
    service, transfer = (3, 5, 4, 2), (1, 1, 1)
    requests = saturated(8, 30)
    _, returned = play(ctx, service, transfer, requests)
    n_tokens = 8 * 30
    floor_ms = n_tokens * max(service)  # the slowest stage must process every token
    ctx.metric("bottleneck_bound", floor_ms, unit="ms")
    check.that(
        returned <= floor_ms / 0.95,
        f"the run took {returned:g} ms: throughput is more than 5 percent below the "
        "bottleneck stage's rate",
        f"{n_tokens} tokens through a stage of {max(service)} ms need at least {floor_ms} ms; "
        f"one token at a time takes {sequential_ms(service, transfer, requests)} ms\n",
    )


# Rejects priority_by_request (sequence 0 would be served first every time) and
# greedy_locks.
@case("a single stage: sequences take turns, oldest ready job first")
def single_stage(ctx):
    play(ctx, (2,), (), [("a", 0, 3), ("b", 0, 3), ("c", 1, 2)])
    play(ctx, (3,), (), [("a", 0, 4), ("b", 3, 2), ("c", 6, 2), ("d", 9, 1)])


# Rejects greedy_locks, ties_to_newcomer, priority_by_request.
@case("requests that arrive together, in bursts, and at the instant a token returns")
def simultaneous_arrivals(ctx):
    play(ctx, (1, 2), (1,), [("a", 0, 2), ("b", 0, 2), ("c", 0, 2), ("d", 0, 2), ("e", 0, 1)])
    play(
        ctx, (2, 1, 2), (1, 1), [("a", 0, 3), ("b", 7, 2), ("c", 7, 2), ("d", 14, 2), ("e", 14, 1)]
    )
    play(ctx, (3, 3), (2,), [("a", 0, 4), ("b", 8, 3), ("c", 16, 3), ("d", 16, 1), ("e", 24, 2)])


# Rejects transfer_inside_next_stage and free_transfers where links are slow.
@case("slow links, a bottleneck in the first stage, and one in the last")
def uneven_stages(ctx):
    play(ctx, (1, 1, 1), (6, 4), saturated(4, 5))
    play(ctx, (9, 1, 2), (1, 1), saturated(5, 6))
    play(ctx, (1, 2, 9), (1, 3), saturated(5, 6))


# Rejects returns_early: the last job ends after the last decision is made.
@case("idle gaps between requests, and a long final token")
def idle_gaps(ctx):
    play(ctx, (2, 7), (3,), [("a", 0, 1), ("b", 40, 2), ("c", 41, 1), ("d", 200, 1)])
    play(ctx, (1, 1, 1, 20), (1, 1, 1), [("a", 5, 2), ("b", 6, 1)])


@case("random pipelines of 1 to 5 stages and up to 10 sequences")
def random_pipelines(ctx):
    rng = ctx.rng
    for _ in range(25):
        n_stages = int(rng.integers(1, 6))
        service = tuple(int(s) for s in rng.integers(1, 9, size=n_stages))
        transfer = tuple(int(s) for s in rng.integers(1, 5, size=n_stages - 1))
        arrivals = sorted(int(a) for a in rng.integers(0, 40, size=int(rng.integers(1, 11))))
        cycle = sum(service) + sum(transfer)
        if rng.random() < 0.5:  # arrivals on the grid of one sequence's cycle: many ties
            arrivals = sorted(int(a) // cycle * cycle for a in arrivals)
        requests = [(f"r{i}", a, int(rng.integers(1, 7))) for i, a in enumerate(arrivals)]
        play(ctx, service, transfer, requests)
