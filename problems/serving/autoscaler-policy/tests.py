"""Tests for autoscaler-policy.

`desired_replicas` is a pure function, so most cases call it directly on scripted
metrics and compare the replica counts with literal values or with the oracle's. Two
cases close the loop: the policy is evaluated once per second against the kit's
`SimCluster`, whose new replicas take four seconds to become ready, on a load step
and on a load that comes in bursts. There the decisions must equal the oracle's, and
the three numbers an autoscaler is judged by (requests that waited too long, replica
time paid for, and how often the size changed) must stay under the statement's
limits.
"""

import copy
from dataclasses import dataclass
from types import SimpleNamespace

from whetstone.judge import CheckFailure, case, check
from whetstone.kit import CostModel, SimCluster, workloads

COST = CostModel(base_s=2**-6, prefill_s_per_token=2**-12, decode_s_per_seq=2**-9)
STARTUP_S, PERIOD_S, SLO_S = 4.0, 1.0, 1.0


@dataclass(frozen=True)
class Config:
    target: int
    tolerance: float
    min_replicas: int
    max_replicas: int
    window_s: float


SIM_CFG = Config(target=6, tolerance=0.25, min_replicas=1, max_replicas=40, window_s=20.0)


def snapshot(now, ready, starting, queue_depth, running):
    return SimpleNamespace(
        now=now, ready=ready, starting=starting, draining=0, queue_depth=queue_depth,
        running=running, outstanding_tokens=40 * (queue_depth + running), kv_usage=0.0,
    )  # fmt: skip


def evaluate(fn, metrics, state, cfg):
    result = fn(metrics, state, cfg)
    if not (isinstance(result, tuple) and len(result) == 2):
        raise CheckFailure("desired_replicas must return a tuple (n, state)")
    n, state = result
    if type(n) is not int:
        raise CheckFailure(f"desired_replicas returned n = {n!r}, which is not an int")
    return n, state


def play(ctx, cfg, rows, label):
    """rows: (now, ready, starting, queue_depth, running, expected n, why)."""
    state = None
    history = [f"{cfg}"]
    for now, ready, starting, queue, running, expected, why in rows:
        n, state = evaluate(
            ctx.solution.desired_replicas,
            snapshot(now, ready, starting, queue, running),
            state,
            cfg,
        )
        history.append(
            f"t={now:g}: ready={ready} starting={starting} queue_depth={queue} "
            f"running={running} -> {n}"
        )
        check.that(
            n == expected,
            f"{label}: at t={now:g} the policy returned {n}, which is not the specified count",
            "\n".join(history) + f"\nexpected: {expected}  ({why})\n",
        )


def play_against_oracle(ctx, cfg, rows, label):
    """rows: (now, ready, starting, queue_depth, running). Compares with the oracle."""
    mine = theirs = None
    history = [f"{cfg}"]
    for now, ready, starting, queue, running in rows:
        metrics = snapshot(now, ready, starting, queue, running)
        got, mine = evaluate(ctx.solution.desired_replicas, metrics, mine, cfg)
        want, theirs = evaluate(ctx.oracle.desired_replicas, metrics, theirs, cfg)
        history.append(
            f"t={now:g}: ready={ready} starting={starting} queue_depth={queue} "
            f"running={running} -> {got}"
        )
        check.that(
            got == want,
            f"{label}: at t={now:g} the policy returned {got}, which is not the specified count",
            "\n".join(history[-14:]) + f"\nexpected: {want}\n",
        )


EXAMPLE_CFG = Config(target=10, tolerance=0.25, min_replicas=1, max_replicas=8, window_s=30.0)
EXAMPLE = [
    (0, 2, 0, 0, 14, 2, "load 14 on 2 replicas: ceil(14 / 10) = 2"),
    (10, 2, 0, 41, 16, 6, "load 57: ceil(57 / 10) = 6, a scale-up, applied at once"),
    (20, 2, 4, 36, 16, 6, "6 replicas count (4 are starting); load 52 is within 25% of 60"),
    (30, 6, 0, 0, 21, 6, "recommends 3, but the window (10, 30] still holds a 6"),
    (40, 6, 0, 0, 18, 6, "recommends 2; the window still holds the 6 of t=20"),
    (50, 6, 0, 0, 44, 5, "recommends 5; the window holds 3, 2, 5: scale down to 5"),
    (60, 5, 0, 0, 12, 5, "recommends 2; the window holds 2, 5, 2"),
    (70, 5, 0, 0, 9, 5, "recommends 1; the window holds 5, 2, 1"),
    (80, 5, 0, 0, 8, 2, "recommends 1; the window holds 2, 1, 1: scale down to 2"),
]


@case("the worked example: scale up at once, count starting replicas, scale down late",
      public=True)  # fmt: skip
def worked_example(ctx):
    # Rejects no_stabilization (t=30), pending_ignored (t=20), window_inclusive (t=30,
    # where the 2 of t=0 would not matter but the 6 of t=10 at t=40 does), window_mean.
    play(ctx, EXAMPLE_CFG, EXAMPLE, "the worked example")


@case("the proportional rule: ceiling, the tolerance band, the bounds", public=True)
def proportional_rule(ctx):
    # Rejects floor_division, no_tolerance, min_ignored, ignores_queue.
    cfg = Config(target=8, tolerance=0.25, min_replicas=2, max_replicas=10, window_s=0.0)
    for ready, starting, queue, running, expected, why in (
        (4, 0, 0, 32, 4, "exactly on target"),
        (4, 0, 0, 40, 4, "40 against 32: 8 over, and 0.25 * 32 = 8 is still tolerated"),
        (4, 0, 0, 41, 6, "41 against 32 is outside the band: ceil(41 / 8) = 6"),
        (4, 0, 0, 24, 4, "24 against 32: 8 under is still tolerated"),
        (4, 0, 0, 23, 3, "23 is outside the band: ceil(23 / 8) = 3"),
        (4, 0, 30, 32, 8, "the queue counts: load 62, ceil(62 / 8) = 8"),
        (4, 0, 200, 32, 10, "ceil(232 / 8) = 29, capped at max_replicas"),
        (4, 0, 0, 0, 2, "no load: 0 replicas would do, raised to min_replicas"),
        (4, 0, 0, 9, 2, "ceil(9 / 8) = 2"),
        (2, 3, 0, 17, 3, "5 replicas count; 17 against 40 is outside the band: 3"),
        (1, 0, 5, 8, 2, "13 against 8 is outside the band: ceil(13 / 8) = 2"),
        (3, 0, 1, 24, 3, "25 against 24 is inside the band, although ceil(25 / 8) = 4"),
    ):
        play(ctx, cfg, [(0, ready, starting, queue, running, expected, why)],
             "window_s = 0, one evaluation")  # fmt: skip


@case("the window: the highest recent recommendation, and when one stops being recent",
      public=True)  # fmt: skip
def window_rule(ctx):
    # Rejects window_inclusive, history_never_pruned, window_mean, stabilizes_scale_up.
    cfg = Config(target=4, tolerance=0.0, min_replicas=1, max_replicas=20, window_s=3.0)
    play(ctx, cfg, [
        (0.0, 1, 0, 0, 20, 5, "ceil(20 / 4) = 5"),
        (1.0, 5, 0, 0, 4, 5, "recommends 1; the 5 of t=0 is 1 s old"),
        (2.0, 5, 0, 0, 12, 5, "recommends 3; the 5 of t=0 is 2 s old"),
        (3.0, 5, 0, 0, 4, 3, ("the 5 of t=0 is exactly 3 s old and no longer counts: "
                             "the window holds 1, 3, 1")),
        (4.0, 3, 0, 0, 4, 3, "the window holds 3, 1, 1"),
        (5.0, 3, 0, 0, 4, 1, "the 3 of t=2 is 3 s old: the window holds 1, 1, 1"),
        (5.5, 1, 0, 0, 30, 8, "a scale-up does not wait: ceil(30 / 4) = 8"),
        (6.0, 1, 7, 0, 30, 8, "the 7 starting replicas count: 30 against 32, recommends 8"),
        (8.0, 8, 0, 0, 8, 8, "recommends 2; the 8s of t=5.5 and t=6 are in the window"),
        (8.5, 8, 0, 0, 8, 8, "the 8 of t=5.5 is exactly 3 s old; the 8 of t=6 is not"),
        (9.0, 8, 0, 0, 8, 2, "the window holds 2, 2, 2"),
    ], "window_s = 3")  # fmt: skip


def run_cluster(ctx, fn, requests, cfg, *, horizon_s):
    """Evaluate the policy every PERIOD_S against a SimCluster. Returns what happened."""
    sim = ctx.simclock()
    cluster = SimCluster(sim, COST, replicas=cfg.min_replicas, max_seqs=8, startup_s=STARTUP_S)
    decisions = []

    async def control():
        state = None
        while True:
            metrics = cluster.metrics()
            n, state = evaluate(fn, metrics, state, cfg)
            if n < 0:
                raise CheckFailure(f"desired_replicas returned {n}, a negative replica count")
            decisions.append((metrics.now, metrics.ready + metrics.starting, n))
            cluster.scale_to(n)
            await sim.sleep(PERIOD_S)

    sim.spawn(control())
    workloads.drive(sim, requests, cluster.submit)
    sim.run(until=horizon_s)
    arrived = {e["id"]: e.time for e in sim.trace.of("arrive")}
    first = {e["id"]: e.time for e in sim.trace.of("first_token")}
    late = sum(1 for i, t in arrived.items() if i not in first or first[i] - t > SLO_S)
    return SimpleNamespace(
        decisions=decisions, late=late, failed=cluster.counters.failed,
        replica_seconds=cluster.replica_seconds(),
        actions=sum(1 for _, current, n in decisions if n != current),
        peak=max(n for _, _, n in decisions),
    )  # fmt: skip


def check_cluster(ctx, requests, *, horizon_s, max_late, max_replica_seconds, max_actions,
                  max_peak):  # fmt: skip
    want = run_cluster(ctx, ctx.oracle.desired_replicas, requests, SIM_CFG, horizon_s=horizon_s)
    got = run_cluster(ctx, ctx.solution.desired_replicas, requests, SIM_CFG, horizon_s=horizon_s)
    ctx.metric("requests", len(requests))
    check.at_most("failed_requests", got.failed, 0, unit="requests")
    check.at_most("late_requests", got.late, max_late, unit="requests")
    check.at_most("replica_seconds", round(got.replica_seconds, 1), max_replica_seconds)
    check.at_most("scaling_actions", got.actions, max_actions)
    check.at_most("peak_replicas", got.peak, max_peak, unit="replicas")
    wrong = next(
        (i for i, (g, w) in enumerate(zip(got.decisions, want.decisions, strict=False)) if g != w),
        None,
    )
    if wrong is not None:
        now, _, n = got.decisions[wrong]
        lines = [
            f"t={t:g}: {c} replicas -> {m}"
            for t, c, m in got.decisions[max(0, wrong - 8) : wrong + 1]
        ]
        check.that(
            False,
            f"at t={now:g} the policy returned {n}, which is not the specified count",
            "\n".join(lines) + f"\nexpected at t={now:g}: {want.decisions[wrong][2]}\n",
        )


@case("a load step from 5 to 30 requests/s and back: late requests, replica-seconds, actions",
      public=True)  # fmt: skip
def load_step(ctx):
    # Rejects ignores_queue and stabilizes_scale_up (late requests), history_never_pruned
    # (replica-seconds), no_stabilization (late requests and actions).
    requests = workloads.phases(ctx.rng, [(20.0, 5.0), (40.0, 30.0), (40.0, 5.0)],
                                prompt_tokens=(8, 64), output_tokens=(4, 32))  # fmt: skip
    check_cluster(ctx, requests, horizon_s=120.0, max_late=160, max_replica_seconds=800,
                  max_actions=18, max_peak=22)  # fmt: skip


@case("six bursts of 6 s, 6 s apart: the replicas must survive the gaps", public=True)
def bursts(ctx):
    # Rejects no_stabilization (every burst waits for cold starts), pending_ignored
    # (overshoot while replicas start), window_mean.
    phases = [(10.0, 5.0)] + [(6.0, 30.0), (6.0, 5.0)] * 6 + [(30.0, 5.0)]
    requests = workloads.phases(ctx.rng, phases, prompt_tokens=(8, 64), output_tokens=(4, 32))
    check_cluster(ctx, requests, horizon_s=132.0, max_late=250, max_replica_seconds=900,
                  max_actions=18, max_peak=22)  # fmt: skip


@case("the policy is a function of its arguments: replaying a state gives the same answer")
def pure_function(ctx):
    # Rejects history_mutated.
    cfg = Config(target=5, tolerance=0.125, min_replicas=1, max_replicas=12, window_s=4.0)
    fn = ctx.solution.desired_replicas
    state = None
    trail = []
    for step in range(12):
        load = [3, 40, 44, 12, 9, 8, 30, 2, 2, 2, 2, 2][step]
        metrics = snapshot(float(step), max(1, load // 9), 0, max(0, load - 8), min(load, 8))
        saved = copy.deepcopy(state)
        first, next_state = evaluate(fn, metrics, state, cfg)
        check.that(
            state == saved,
            "desired_replicas modified the state it was given instead of returning a new one",
        )
        again, _ = evaluate(fn, metrics, state, cfg)
        check.that(
            again == first,
            "calling desired_replicas twice with the same arguments gave two different counts",
            f"at t={step}: first {first}, then {again}\n",
        )
        trail.append((metrics, state, first))
        state = next_state
    for metrics, old_state, expected in trail:  # earlier states are still valid inputs
        again, _ = evaluate(fn, metrics, old_state, cfg)
        check.that(
            again == expected,
            "an earlier state, passed in again later, gave a different count than before",
        )


@case("random scripts: irregular evaluation times, replicas that start, bounds that bind")
def random_scripts(ctx):
    rng = ctx.rng
    for trial in range(40):
        cfg = Config(
            target=int(rng.integers(1, 12)),
            tolerance=float(rng.choice([0.0, 0.125, 0.25, 0.5])),
            min_replicas=int(rng.integers(1, 4)),
            max_replicas=int(rng.integers(4, 20)),
            window_s=float(rng.choice([0.0, 1.0, 2.5, 6.0])),
        )
        now, rows = 0.0, []
        for _ in range(50):
            ready = int(rng.integers(1, 20))
            starting = int(rng.integers(0, 4)) if rng.random() < 0.3 else 0
            level = int(rng.choice([0, 1, 3, 10, 30]))
            running = int(rng.integers(0, ready * 8 + 1))
            queue = int(rng.integers(0, level * cfg.target + 1))
            rows.append((now, ready, starting, queue, running))
            now += float(rng.choice([0.5, 1.0, 1.0, 1.5, 3.0]))
        play_against_oracle(ctx, cfg, rows, f"script {trial}")


@case("closed loop by hand: the count follows the policy's own previous answers")
def closed_loop_script(ctx):
    rng = ctx.rng
    for trial in range(20):
        cfg = Config(target=int(rng.integers(2, 9)), tolerance=0.25, min_replicas=1,
                     max_replicas=24, window_s=float(rng.choice([3.0, 5.0, 8.0])))  # fmt: skip
        mine = theirs = None
        current, starting, load = 1, 0, 0
        for step in range(80):
            if step % 7 == 0:
                load = int(rng.choice([0, 2, 5, 20, 60, 150]))
            load = max(0, load + int(rng.integers(-3, 4)))
            ready = current - starting
            running = min(load, ready * 8)
            metrics = snapshot(float(step), ready, starting, load - running, running)
            got, mine = evaluate(ctx.solution.desired_replicas, metrics, mine, cfg)
            want, theirs = evaluate(ctx.oracle.desired_replicas, metrics, theirs, cfg)
            check.that(
                got == want,
                f"script {trial}: at t={step} the policy returned {got}, which is not the "
                "specified count",
                f"{cfg}\nready={ready} starting={starting} queue_depth={load - running} "
                f"running={running}\nexpected: {want}\n",
            )
            starting = max(0, want - current) if want > current else 0
            current = want
