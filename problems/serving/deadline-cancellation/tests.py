"""Tests for deadline-cancellation.

`simulate` plays the clients: at each arrival it calls `scheduler.submit(request)` and
starts a task that awaits `handle.result()` and records how the request ended; it
calls `handle.cancel()` at the scripted instants. The engine is a `SimEngine` on a
simulated clock whose unit is the millisecond.

The statement determines everything, so each case checks the outcomes (when, how, how
many tokens), the engine steps, and the releases: against literal values on the
hand-checked scenarios, and on seeded workloads against rules computed here from the
statement and against the oracle.
"""

import dataclasses

from whetstone.judge import case, check
from whetstone.kit import CostModel, ProtocolViolation, SimEngine, SimRequest, workloads

# A step over P prompt tokens and D decoded sequences takes 8 + 0.25 * P + D milliseconds.
COST = CostModel(base_s=8, prefill_s_per_token=0.25, decode_s_per_seq=1)


class Run:
    """The outcome of one simulation."""

    def __init__(self, sim, engine, requests, cancels, max_running):
        self.sim = sim
        self.engine = engine
        self.requests = requests
        self.cancels = cancels
        self.max_running = max_running

    @property
    def steps(self):
        """[(start time, prefill, decode)]."""
        return [(e.time, tuple(e["prefill"]), tuple(e["decode"])) for e in self.sim.trace.of("step")]

    @property
    def releases(self):
        """[(time, id)] of `engine.release` calls."""
        return [(e.time, e["id"]) for e in self.sim.trace.of("release")]

    @property
    def outcomes(self):
        """{id: (time, how, tokens)}; how is "finished", "canceled", or "expired"."""
        return {e["id"]: (e.time, e["how"], e["tokens"]) for e in self.sim.trace.of("outcome")}

    @property
    def cancel_results(self):
        """[(time, id, what cancel() returned)]."""
        return [(e.time, e["id"], e["returned"]) for e in self.sim.trace.of("cancel")]

    def timeline(self):
        """A readable log of the simulation, for failure details."""
        lines = [f"max_running={self.max_running}; a step takes 8 + 0.25 * P + D ms", "timeline:"]
        kinds = ("arrive", "step", "token", "finish", "release", "cancel", "outcome")
        shown = [e for e in self.sim.trace.events if e.kind in kinds]
        for e in shown[:90]:
            if e.kind == "arrive":
                r = e["request"]
                what = f"arrive   {r.id!r} (prompt {r.prompt_tokens}, output {r.output_tokens}"
                what += f", deadline {r.deadline_s:g})" if r.deadline_s is not None else ")"
            elif e.kind == "step":
                what = f"step {e['index']}: prefill={list(e['prefill'])} decode={list(e['decode'])}"
                what += f"  ({e['duration']:g} ms)"
            elif e.kind == "token":
                what = f"  token  {e['id']!r} #{e['index']}"
            elif e.kind == "cancel":
                what = f"cancel() {e['id']!r} -> {e['returned']!r}"
            elif e.kind == "outcome":
                what = f"client   {e['id']!r}: {e['how']}"
                what += f" with {e['tokens']} tokens" if e["how"] == "finished" else ""
            else:
                what = f"{e.kind:<8} {e['id']!r}"
            lines.append(f"  t={e.time:>8g}  {what}")
        if len(shown) > 90:
            lines.append(f"  ... {len(shown) - 90} more events")
        return "\n".join(lines) + "\n"


def horizon(requests, cancels):
    """A time by which any scheduler that keeps working has ended every request."""
    serial = sum(
        COST.step_s(r.prompt_tokens, 0) + r.output_tokens * COST.step_s(0, 1) for r in requests
    )
    return max([requests[-1].arrival_s, *cancels.values()]) + 2 * serial + 1000


def simulate(ctx, module, requests, cancels=None, *, max_running):
    """Run `module.Scheduler` over `requests`; `cancels` maps a request id to the
    instant at which its client calls `handle.cancel()`."""
    cancels = cancels or {}
    sim = ctx.simclock()
    engine = SimEngine(sim, COST, max_seqs=max_running)
    scheduler = module.Scheduler(engine, max_running)
    run = Run(sim, engine, requests, cancels, max_running)
    handles = {}

    async def client(request, handle):
        try:
            tokens = await handle.result()
            how = "finished"
        except module.Canceled:
            tokens, how = None, "canceled"
        except module.DeadlineExceeded:
            tokens, how = None, "expired"
        sim.trace.record("outcome", id=request.id, how=how, tokens=tokens)

    def arrive(request):
        handles[request.id] = scheduler.submit(request)
        sim.spawn(client(request, handles[request.id]))

    def cancel(request_id):
        sim.trace.record("cancel", id=request_id, returned=handles[request_id].cancel())

    workloads.drive(sim, requests, arrive)
    # Scheduled after the arrivals: a cancel at the arrival instant comes after `submit`.
    for request_id, when in cancels.items():
        sim.call_at(when, cancel, request_id)
    sim.spawn(scheduler.run())
    try:
        sim.run(until=horizon(requests, cancels))
    except ProtocolViolation:
        pass  # reported below, with the timeline
    if engine.violations:
        first = engine.violations[0]
        check.that(
            False,
            f"the scheduler broke the engine's protocol ({first.code})",
            f"{first.code}: {first}\n" + run.timeline(),
        )
    pending = [r.id for r in requests if r.id not in run.outcomes]
    check.that(
        not pending,
        f"{len(pending)} of {len(requests)} clients are still waiting for `result()` "
        "at the end of the simulation",
        f"no outcome for: {pending[:10]!r}\n" + run.timeline(),
    )
    return run


def check_rules(run):
    """The statement, restated as properties of the trace (independent of the oracle)."""
    detail = run.timeline()
    finish_time = {e["id"]: e.time for e in run.sim.trace.of("finish")}
    last_token = {}
    for e in run.sim.trace.of("token"):
        last_token[e["id"]] = e.time
    called = {}
    for when, request_id, returned in run.cancel_results:
        check.that(type(returned) is bool, "cancel() must return True or False", detail)
        called[request_id] = (when, returned)
    for r in run.requests:
        when, how, tokens = run.outcomes[r.id]
        who = f"request {r.id!r}: "
        cancel_at = run.cancels.get(r.id)
        if how == "finished":
            check.that(r.id in finish_time, "a client was told its request finished, but "
                       "the engine never finished it", who + "\n" + detail)  # fmt: skip
            check.that(when == finish_time[r.id], "result() did not return at the instant of "
                       "the last token", who + "\n" + detail)  # fmt: skip
            check.that(tokens == r.output_tokens, "result() returned the wrong token count",
                       f"{who}returned {tokens!r}\n" + detail)  # fmt: skip
            check.that(r.deadline_s is None or when < r.deadline_s, "a request finished at or "
                       "after its deadline", who + "\n" + detail)  # fmt: skip
            check.that(cancel_at is None or when < cancel_at, "a request finished at or after "
                       "the instant it was canceled", who + "\n" + detail)  # fmt: skip
        else:
            end = cancel_at if how == "canceled" else max(r.deadline_s or 0, r.arrival_s)
            check.that(
                (how == "canceled" and cancel_at is not None)
                or (how == "expired" and r.deadline_s is not None),
                "a client received an exception for something that did not happen",
                f"{who}{how}\n" + detail,
            )
            check.that(
                when == end,
                "a client was not told at the instant its request was canceled or expired",
                f"{who}{how} at t={end:g}, client resumed at t={when:g}\n" + detail,
            )
            check.that(
                r.id not in last_token or last_token[r.id] < end,
                "the engine produced a token for a request at or after the instant it "
                "was canceled or expired",
                f"{who}{how} at t={end:g}, last token at t={last_token.get(r.id, 0):g}\n" + detail,
            )
        if r.id in called:
            check.that(
                called[r.id][1] == (how == "canceled"),
                "cancel() returned the wrong value",
                f"{who}{how}; cancel() returned {called[r.id][1]!r}\n" + detail,
            )
    check.that(
        not run.engine.live and not run.engine.busy,
        "the engine still holds sequences at the end of the simulation",
        f"live: {run.engine.live!r}\n" + detail,
    )


def expect(run, *, outcomes=None, steps=None, releases=None):
    """Compare with literal values, without leaking them in the message."""
    detail = run.timeline()
    if outcomes is not None:
        got = run.outcomes
        for request_id, want in outcomes.items():
            have = got[request_id]
            line = f"request {request_id!r}: expected (time, how, tokens) {want!r}, got {have!r}\n"
            check.that(have[1] == want[1], "a request ended the wrong way", line + detail)
            check.that(have[0] == want[0], "a client got its result at the wrong time",
                       line + detail)  # fmt: skip
            check.that(have[2] == want[2], "result() returned the wrong token count",
                       line + detail)  # fmt: skip
    if releases is not None:
        # Releases at one instant may come in any order.
        def canonical(pairs):
            return sorted((when, repr(seq_id)) for when, seq_id in pairs)

        check.that(
            canonical(run.releases) == canonical(releases),
            "engine.release was not called for the right sequences at the right times",
            f"expected (time, id): {releases!r}\ngot: {run.releases!r}\n" + detail,
        )
    if steps is not None:
        got = run.steps
        lines = "expected steps (start ms, prefill, decode):\n"
        lines += "".join(f"  {step!r}\n" for step in steps[:40])
        lines += "got:\n" + "".join(f"  {step!r}\n" for step in got[:40]) + detail
        for k, (have, want) in enumerate(zip(got, steps, strict=False)):
            check.that(have[1] == want[1], f"step {k} prefills the wrong sequences", lines)
            check.that(have[2] == want[2], f"step {k} decodes the wrong sequences", lines)
            check.that(have[0] == want[0], f"step {k} starts at the wrong time", lines)
        check.that(len(got) == len(steps), "the number of engine steps is wrong", lines)


def solve(ctx, requests, cancels=None, *, max_running):
    run = simulate(ctx, ctx.solution, requests, cancels, max_running=max_running)
    check_rules(run)
    return run


def matches_oracle(ctx, requests, cancels=None, *, max_running):
    got = solve(ctx, requests, cancels, max_running=max_running)
    want = simulate(ctx, ctx.oracle, requests, cancels, max_running=max_running)
    expect(got, outcomes=want.outcomes, releases=want.releases, steps=want.steps)
    for name in ("steps", "prefill_tokens", "decode_tokens", "output_tokens"):
        check.that(
            getattr(got.engine.counters, name) == getattr(want.engine.counters, name),
            f"the engine's total work ({name}) differs from the reference",
            got.timeline(),
        )
    return got


def req(id, arrival, prompt, output, deadline=None):
    return SimRequest(id, arrival, prompt, output, deadline_s=deadline)


def with_deadlines_and_cancels(rng, requests, *, deadline_ms, cancel_ms):
    """Give about half of the requests a deadline (a whole number of ms after arrival)
    and cancel about a quarter (at arrival + k + 0.5 ms, so never on a deadline)."""
    out, cancels = [], {}
    for r in requests:
        if rng.random() < 0.5:
            r = dataclasses.replace(r, deadline_s=r.arrival_s + int(rng.integers(*deadline_ms)))
        if rng.random() < 0.25:
            cancels[r.id] = r.arrival_s + int(rng.integers(*cancel_ms)) + 0.5
        out.append(r)
    return out, cancels


@case("the worked example: a cancel and three deadlines, waiting and running", public=True)
def worked_example(ctx):
    requests = [req(0, 0, 8, 6), req(1, 0, 8, 3, 30), req(2, 4, 8, 2, 25),
                req(3, 6, 16, 3, 70), req(4, 12, 8, 2, 56)]
    run = solve(ctx, requests, {0: 36}, max_running=2)
    expect(
        run,
        outcomes={0: (36, "canceled", None), 1: (30, "expired", None), 2: (25, "expired", None),
                  3: (65, "finished", 3), 4: (56, "expired", None)},
        releases=[(30, 1), (36, 0), (56, 4)],
        steps=[
            (0, ((0, 8), (1, 8)), ()),
            (12, (), (0, 1)),
            (22, (), (0, 1)),
            (32, ((3, 16),), (0,)),
            (45, ((4, 8),), (3,)),
            (56, (), (3,)),
        ],
    )  # fmt: skip


@case("without cancels or deadlines it is plain continuous batching", public=True)
def nothing_to_cancel(ctx):
    requests = [req(0, 0, 8, 4), req(1, 0, 16, 2), req(2, 5, 8, 3, 1000)]
    run = solve(ctx, requests, max_running=2)
    expect(
        run,
        outcomes={0: (45, "finished", 4), 1: (24, "finished", 2), 2: (54, "finished", 3)},
        releases=[],
        steps=[(0, ((0, 8), (1, 16)), ()), (14, (), (0, 1)), (24, ((2, 8),), (0,)),
               (35, (), (0, 2)), (45, (), (2,))],
    )  # fmt: skip


@case("canceling a waiting request: it is never admitted", public=True)
def cancel_while_waiting(ctx):
    requests = [req("a", 0, 8, 3), req("b", 1, 8, 2), req("c", 2, 8, 2)]
    run = solve(ctx, requests, {"b": 15}, max_running=1)
    expect(
        run,
        outcomes={"a": (28, "finished", 3), "b": (15, "canceled", None), "c": (47, "finished", 2)},
        releases=[],
        steps=[(0, (("a", 8),), ()), (10, (), ("a",)), (19, (), ("a",)), (28, (("c", 8),), ()),
               (38, (), ("c",))],
    )  # fmt: skip


@case("canceling a running request mid-step: no more tokens, and the slot is reused",
      public=True)
def cancel_while_running(ctx):
    # Rejects cancel_only_while_waiting and release_deferred_to_step_boundary: "a" is
    # canceled at t=14, during the step that runs from 10 to 19.
    requests = [req("a", 0, 8, 5), req("b", 1, 8, 2)]
    run = solve(ctx, requests, {"a": 14}, max_running=1)
    expect(
        run,
        outcomes={"a": (14, "canceled", None), "b": (38, "finished", 2)},
        releases=[(14, "a")],
        steps=[(0, (("a", 8),), ()), (10, (), ("a",)), (19, (("b", 8),), ()), (29, (), ("b",))],
    )


@case("a deadline expires a running request at that instant, not at the next step",
      public=True)
def deadline_while_running(ctx):
    # Rejects expiry_only_at_admission.
    requests = [req("a", 0, 8, 5, 21.5), req("b", 1, 8, 2, 100)]
    run = solve(ctx, requests, max_running=1)
    expect(
        run,
        outcomes={"a": (21.5, "expired", None), "b": (47, "finished", 2)},
        releases=[(21.5, "a")],
        steps=[(0, (("a", 8),), ()), (10, (), ("a",)), (19, (), ("a",)), (28, (("b", 8),), ()),
               (38, (), ("b",))],
    )  # fmt: skip


@case("a token due at exactly the deadline is not produced; one instant earlier it is")
def deadline_at_the_end_of_a_step(ctx):
    # Rejects deadline_by_sleeping_task. The last token of "a" is due at t=19.
    run = solve(ctx, [req("a", 0, 8, 2, 19)], max_running=2)
    expect(run, outcomes={"a": (19, "expired", None)}, releases=[(19, "a")],
           steps=[(0, (("a", 8),), ()), (10, (), ("a",))])  # fmt: skip
    run = solve(ctx, [req("a", 0, 8, 2, 19.25)], max_running=2)
    expect(run, outcomes={"a": (19, "finished", 2)}, releases=[])
    # The same tie for cancel(): called at the instant the last token is due.
    run = solve(ctx, [req("a", 0, 8, 2)], {"a": 19}, max_running=2)
    expect(run, outcomes={"a": (19, "canceled", None)}, releases=[(19, "a")])


@case("cancel() after the end returns False and changes nothing; so does a second cancel()")
def cancel_too_late(ctx):
    # Rejects stale_deadline_timer: "a" finishes at 19, long before its deadline at 60,
    # and "c" is still running when that deadline comes.
    requests = [req("a", 0, 8, 2, 60), req("b", 1, 8, 2, 12), req("c", 2, 8, 8)]
    sim_run = simulate(ctx, ctx.solution, requests, {"a": 25, "b": 20}, max_running=1)
    check_rules(sim_run)
    expect(
        sim_run,
        outcomes={"a": (19, "finished", 2), "b": (12, "expired", None), "c": (92, "finished", 8)},
        releases=[],
    )
    check.equal(sim_run.cancel_results, [(20, "b", False), (25, "a", False)],
                label="what cancel() returned")  # fmt: skip
    # Two cancels of the same running request: the first wins, the second is a no-op.
    sim = ctx.simclock()
    engine = SimEngine(sim, COST, max_seqs=1)
    scheduler = ctx.solution.Scheduler(engine, 1)
    returned = []
    sim.call_at(0, lambda: returned.append(scheduler.submit(req("x", 0, 8, 4))))
    sim.call_at(12, lambda: returned.append(returned[0].cancel()))
    sim.call_at(13, lambda: returned.append(returned[0].cancel()))
    sim.spawn(scheduler.run())
    sim.run(until=200)
    check.equal(returned[1:], [True, False], label="two cancel() calls")
    check.that(not engine.violations, "the second cancel() broke the engine's protocol")
    check.equal([e["id"] for e in sim.trace.of("release")], ["x"], label="released sequences")


@case("a request whose deadline has already passed when it arrives is never admitted")
def deadline_already_passed(ctx):
    # Rejects late_arrival_admitted. "late" arrives at t=5 with deadline 5; "old" with 3.
    requests = [req("late", 5, 8, 2, 5), req("old", 5, 8, 2, 3), req("ok", 5, 8, 2, 40)]
    run = solve(ctx, requests, max_running=2)
    expect(
        run,
        outcomes={"late": (5, "expired", None), "old": (5, "expired", None),
                  "ok": (24, "finished", 2)},
        releases=[],
        steps=[(5, (("ok", 8),), ()), (15, (), ("ok",))],
    )  # fmt: skip


@case("everything running is canceled mid-step: the step ends empty and the engine idles")
def all_canceled(ctx):
    requests = [req("a", 0, 8, 4), req("b", 0, 8, 4, 15), req("c", 3, 8, 4), req("d", 90, 8, 1)]
    run = solve(ctx, requests, {"a": 15, "c": 9}, max_running=2)
    expect(
        run,
        outcomes={"a": (15, "canceled", None), "b": (15, "expired", None),
                  "c": (9, "canceled", None), "d": (100, "finished", 1)},
        releases=[(15, "b"), (15, "a")],
        steps=[(0, (("a", 8), ("b", 8)), ()), (12, (), ("a", "b")), (90, (("d", 8),), ())],
    )  # fmt: skip


@case("a slot freed by a cancel or a deadline goes to the oldest waiting request")
def slots_are_not_leaked(ctx):
    # Rejects slot_leaked: more requests are removed than there are slots.
    requests = [req(i, 2 * i, 8, 6, 40 + 2 * i if i % 3 == 0 else None) for i in range(12)]
    cancels = {1: 30.5, 2: 44.5, 4: 51.5, 7: 20.5, 8: 120.5}
    matches_oracle(ctx, requests, cancels, max_running=2)


@case("random traffic with deadlines and cancels: outcomes, steps, and engine work")
def random_workloads(ctx):
    for max_running, rate in ((1, 0.01), (3, 0.03), (6, 0.05), (4, 0.05)):
        requests = workloads.poisson(ctx.rng, rate=rate, n=100, prompt_tokens=(4, 96),
                                     output_tokens=(1, 24), quantum_s=1)
        requests, cancels = with_deadlines_and_cancels(ctx.rng, requests, deadline_ms=(0, 600),
                                                       cancel_ms=(0, 400))
        matches_oracle(ctx, requests, cancels, max_running=max_running)


@case("bursts: deadlines that fall exactly when steps end")
def bursty_workloads(ctx):
    for max_running in (2, 4):
        requests = workloads.bursty(ctx.rng, bursts=6, burst_size=6, period_s=128,
                                    prompt_tokens=8, output_tokens=(1, 8), quantum_s=1)
        # With 8-token prompts every step lasts a whole number of milliseconds, so
        # these whole-millisecond deadlines often coincide with the end of a step.
        requests, cancels = with_deadlines_and_cancels(ctx.rng, requests, deadline_ms=(8, 90),
                                                       cancel_ms=(0, 80))
        matches_oracle(ctx, requests, cancels, max_running=max_running)
