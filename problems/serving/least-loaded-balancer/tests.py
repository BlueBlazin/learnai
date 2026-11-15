"""Tests for least-loaded-balancer.

`pick` is checked on hand-computed draws and, statistically, against the exact
distribution that two distinct uniform choices induce. `Balancer` is checked on
scripted reports and routes, and then inside the kit's `SimCluster`: eight replicas, a
seeded Poisson workload, and load reports that arrive only every few seconds. The
routing decisions must equal the oracle's exactly, and the peak load must stay under
the limit the statement gives.
"""

import numpy as np

from whetstone.judge import CheckFailure, case, check, percentile
from whetstone.kit import CostModel, SimCluster, workloads

COST = CostModel(base_s=2**-6, prefill_s_per_token=2**-12, decode_s_per_seq=2**-9)
REPLICAS = 8
PEAK_LIMIT = 1100  # outstanding tokens on one replica; the statement's limit


def check_pick(ctx, replicas, load, u1, u2, expected, label):
    got = ctx.call("pick", list(replicas), dict(load), u1, u2)
    check.that(
        type(got) is int and got == expected,
        f"{label}: returned {got!r}, which is not the specified replica",
        f"replicas = {replicas!r}\nload = {load!r}\nu1 = {u1!r}, u2 = {u2!r}\n"
        f"expected: {expected!r}\n     got: {got!r}\n",
    )


def exact_distribution(replicas, load):
    """P(pick returns replicas[k]) over two distinct uniform draws, by enumeration."""
    n = len(replicas)
    wins = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                second_wins = load[replicas[j]] < load[replicas[i]]
                wins[j if second_wins else i] += 1
    return [w / (n * (n - 1)) for w in wins]


def sample_picks(ctx, replicas, load, n_samples):
    uniforms = ctx.rng.random((n_samples, 2))
    position = {replica: k for k, replica in enumerate(replicas)}
    picks = []
    for u1, u2 in uniforms:
        got = ctx.solution.pick(list(replicas), dict(load), float(u1), float(u2))
        if got not in position:
            raise CheckFailure(f"pick returned {got!r}, which is not one of the replicas")
        picks.append(position[got])
    return picks


@case("the worked example: five replicas, draws that shift, tie, and repeat", public=True)
def worked_example(ctx):
    # Rejects same_replica_twice and second_not_shifted (the shifted draws),
    # ties_to_lower_id and ties_to_second (the two tied draws), random_routing.
    replicas = [0, 1, 2, 3, 4]
    load = {0: 300, 1: 120, 2: 120, 3: 500, 4: 200}
    for u1, u2, expected, what in (
        (0.1, 0.9, 4, "first 0, second 4: 200 < 300"),
        (0.7, 0.1, 0, "first 3, second 0: 300 < 500"),
        (0.5, 0.5, 2, "first 2, second 3 (shifted past 2): 120 < 500"),
        (0.3, 0.3, 1, "first 1, second 2 (shifted past 1): a tie, the first drawn wins"),
        (0.5, 0.3, 2, "first 2, second 1: a tie, the first drawn wins"),
        (0.9, 0.99, 4, "first 4, second 3: 200 < 500"),
        (0.0, 0.0, 1, "first 0, second 1 (shifted past 0): 120 < 300"),
    ):
        check_pick(ctx, replicas, load, u1, u2, expected, f"pick with u1={u1}, u2={u2} ({what})")


@case("one replica, two replicas, and ids that are not 0..n-1 or not sorted", public=True)
def small_and_unsorted(ctx):
    # Rejects sorts_replicas.
    check_pick(ctx, [7], {7: 99}, 0.4, 0.9, 7, "a single replica")
    check_pick(ctx, [3, 9], {3: 10, 9: 5}, 0.2, 0.7, 9, "two replicas, first 3")
    check_pick(ctx, [3, 9], {3: 10, 9: 5}, 0.8, 0.7, 9, "two replicas, first 9")
    check_pick(ctx, [3, 9], {3: 5, 9: 5}, 0.8, 0.0, 9, "two tied replicas, first 9")
    check_pick(ctx, [3, 9], {3: 5, 9: 5}, 0.2, 0.0, 3, "two tied replicas, first 3")
    replicas = [40, 10, 30, 20]
    load = {10: 1, 20: 2, 30: 3, 40: 4}
    check_pick(ctx, replicas, load, 0.0, 0.9, 20, "ids [40, 10, 30, 20], first 40, second 20")
    check_pick(ctx, replicas, load, 0.6, 0.0, 30, "ids [40, 10, 30, 20], first 30, second 40")
    check_pick(ctx, replicas, load, 0.3, 0.4, 10, "ids [40, 10, 30, 20], first 10, second 30")


@case("Balancer: the worked example, estimates between and across reports", public=True)
def balancer_example(ctx):
    # Rejects no_local_accounting (the third route), request_counts, and report_adds
    # (the route after the report).
    balancer = ctx.solution.Balancer([0, 1, 2])
    script = (
        ("route", 100, 0.0, 0.0, 0),  # all estimates 0: first 0, second 1, tie -> 0
        ("route", 50, 0.0, 0.0, 1),  # 0 holds 100, 1 holds 0 -> 1
        ("route", 30, 0.0, 0.0, 1),  # 100 against 50 -> 1 again, now 80
        ("route", 30, 0.4, 0.9, 2),  # first 1 (80), second 2 (0) -> 2
        ("report", {0: 10, 1: 80, 2: 30}),
        ("route", 40, 0.0, 0.9, 0),  # first 0 (10), second 2 (30) -> 0, now 50
        ("route", 40, 0.0, 0.9, 2),  # first 0 (50), second 2 (30) -> 2
    )
    play(script, balancer, "the worked example")


def play(script, balancer, label):
    history = []
    for step in script:
        if step[0] == "report":
            history.append(f"report({step[1]!r})")
            result = balancer.report(dict(step[1]))
            check.that(result is None, f"{label}: report must return None", "\n".join(history))
        else:
            _, tokens, u1, u2, expected = step
            got = balancer.route(tokens, u1, u2)
            history.append(f"route({tokens}, {u1}, {u2}) -> {got!r}")
            check.that(
                type(got) is int and got == expected,
                f"{label}: route returned {got!r}, which is not the specified replica",
                "\n".join(history) + f"\nexpected: {expected!r}\n",
            )


def simulate(ctx, module, requests, uniforms, *, interval_s, replicas=REPLICAS):
    """Route `requests` through a SimCluster; reports every `interval_s` from t = 0."""
    sim = ctx.simclock()
    balancer = module.Balancer(list(range(replicas)))
    routes, peak = [], 0

    def route(request, ready):
        nonlocal peak
        peak = max(peak, max(replica.outstanding_tokens for replica in ready))
        u1, u2 = (float(u) for u in uniforms[request.id])
        chosen = balancer.route(request.prompt_tokens + request.output_tokens, u1, u2)
        if type(chosen) is not int or not 0 <= chosen < replicas:
            raise CheckFailure(f"route returned {chosen!r}, which is not a replica id")
        routes.append(chosen)
        return chosen

    cluster = SimCluster(sim, COST, replicas=replicas, max_seqs=8, route=route)

    def report():
        balancer.report({replica.id: replica.outstanding_tokens for replica in cluster.replicas})

    # A report due at the instant of an arrival is delivered first (priority -1).
    for k in range(int(requests[-1].arrival_s // interval_s) + 1):
        sim.call_at(k * interval_s, report, priority=-1)
    workloads.drive(sim, requests, cluster.submit)
    sim.run()
    check.that(not cluster.violations, "the router misused the cluster")
    ttft = sim.trace.latencies("arrive", "first_token")
    return routes, peak, percentile(ttft, 99)


def check_routes(ctx, requests, uniforms, label, *, interval_s, replicas=REPLICAS):
    want, _, _ = simulate(
        ctx, ctx.oracle, requests, uniforms, interval_s=interval_s, replicas=replicas
    )
    got, peak, p99 = simulate(
        ctx, ctx.solution, requests, uniforms, interval_s=interval_s, replicas=replicas
    )
    wrong = next((i for i, (g, w) in enumerate(zip(got, want, strict=False)) if g != w), None)
    check.that(
        wrong is None and len(got) == len(want),
        f"{label}: a request was routed to a different replica than specified",
        ""
        if wrong is None
        else (
            f"the first difference is request {wrong}: expected replica {want[wrong]}, "
            f"got {got[wrong]}\nroutes of the first 20 requests:\n"
            f"expected: {want[:20]}\n     got: {got[:20]}\n"
        ),
    )
    return peak, p99


@case("eight replicas, reports every 4 s: peak load at most 1100 outstanding tokens",
      public=True)  # fmt: skip
def stale_reports(ctx):
    # The bound: 1100 tokens. Measured on this workload: the reference peaks at 685,
    # uniform random routing (random_routing) at 1847, and a balancer that trusts
    # 4-second-old reports at 2783 (no_local_accounting) and 3111 (request_counts).
    requests = workloads.poisson(
        ctx.rng, rate=40.0, n=800, prompt_tokens=(16, 256), output_tokens=(8, 64), quantum_s=2**-10
    )
    uniforms = ctx.rng.random((len(requests), 2))
    got, peak, p99 = simulate(ctx, ctx.solution, requests, uniforms, interval_s=4.0)
    check.that(len(got) == len(requests), "not every request was routed")
    ctx.metric("p99_ttft", p99 * 1000, unit="ms")
    check.at_most("peak_outstanding_tokens", peak, PEAK_LIMIT, unit="tokens")


@case("the same cluster: every routing decision equals the specified one", public=True)
def exact_routes(ctx):
    requests = workloads.poisson(
        ctx.rng, rate=30.0, n=400, prompt_tokens=(16, 256), output_tokens=(8, 64), quantum_s=2**-10
    )
    uniforms = ctx.rng.random((len(requests), 2))
    for interval_s in (0.5, 4.0):
        check_routes(
            ctx, requests, uniforms, f"reports every {interval_s:g} s", interval_s=interval_s
        )


@case("choice frequencies with distinct loads and one tied pair follow the exact distribution")
def frequencies(ctx):
    # Rejects same_replica_twice and second_not_shifted (the most loaded replica must
    # never win), ties_to_lower_id and ties_to_second (the tied pair must split evenly).
    replicas = [0, 1, 2, 3, 4]
    load = {0: 300, 1: 120, 2: 120, 3: 500, 4: 200}
    probs = exact_distribution(replicas, load)  # [0.1, 0.35, 0.35, 0.0, 0.2]
    picks = sample_picks(ctx, replicas, load, 20_000)
    check.distribution(picks, probs, label="pick over five replicas")


@case("choice frequencies when every replica is idle: uniform, no replica starved")
def idle_frequencies(ctx):
    # Rejects ties_to_lower_id: with ties to the lower id the highest id is never chosen.
    replicas = [5, 2, 9, 4]
    load = dict.fromkeys(replicas, 0)
    picks = sample_picks(ctx, replicas, load, 20_000)
    check.distribution(picks, [0.25] * 4, label="pick over four idle replicas")
    replicas = list(range(16))
    load = {replica: 7 * (replica % 3) for replica in replicas}
    picks = sample_picks(ctx, replicas, load, 20_000)
    check.distribution(
        picks, exact_distribution(replicas, load), label="pick over sixteen replicas in three tiers"
    )


@case("pick: extreme uniforms, every ordered pair of four replicas")
def every_pair(ctx):
    replicas = [11, 22, 33, 44]
    load = {11: 5, 22: 5, 33: 1, 44: 9}
    n = len(replicas)
    for i in range(n):
        for j in range(n - 1):
            u1, u2 = (i + 0.5) / n, (j + 0.5) / (n - 1)
            expected = ctx.expected("pick", list(replicas), dict(load), u1, u2)
            check_pick(ctx, replicas, load, u1, u2, expected, f"pick at draw ({i}, {j})")
    top = float(np.nextafter(1.0, 0.0))
    for u1, u2 in ((0.0, 0.0), (top, top), (0.0, top), (top, 0.0), (0.25, 1 / 3), (0.75, 2 / 3)):
        expected = ctx.expected("pick", list(replicas), dict(load), u1, u2)
        check_pick(ctx, replicas, load, u1, u2, expected, "pick at the edge of an interval")


@case("Balancer: before any report, repeated reports, a report that lowers and raises loads")
def balancer_scripts(ctx):
    # Rejects report_adds, no_local_accounting, and request_counts.
    rng = ctx.rng
    for trial in range(20):
        ids = [int(v) for v in rng.permutation(rng.choice(range(50), size=5, replace=False))]
        mine, theirs = ctx.solution.Balancer(list(ids)), ctx.oracle.Balancer(list(ids))
        history = []
        for _ in range(60):
            if rng.random() < 0.15:
                loads = {replica: int(rng.integers(0, 3000)) for replica in ids}
                mine.report(dict(loads))
                theirs.report(dict(loads))
                history.append(f"report({loads!r})")
                continue
            tokens = int(rng.choice([1, 20, 300, 4000]))
            u1, u2 = float(rng.random()), float(rng.random())
            got, want = mine.route(tokens, u1, u2), theirs.route(tokens, u1, u2)
            history.append(f"route({tokens}, {u1!r}, {u2!r}) -> {got!r}")
            check.that(
                got == want,
                f"Balancer script {trial}: route returned {got!r}, which is not the specified replica",
                f"Balancer({ids!r})\n" + "\n".join(history[-12:]) + f"\nexpected: {want!r}\n",
            )


@case("bursts that arrive at the instant of a report, and a few very large requests")
def bursts_and_giants(ctx):
    # A report due at the instant of an arrival is delivered first. Rejects
    # request_counts: the large requests weigh as much as fifty small ones.
    bursts = workloads.bursty(
        ctx.rng, bursts=12, burst_size=20, period_s=2.0, prompt_tokens=(16, 200),
        output_tokens=(4, 48),
    )  # fmt: skip
    uniforms = ctx.rng.random((len(bursts), 2))
    check_routes(ctx, bursts, uniforms, "bursts on the report instants", interval_s=2.0)

    mixed = workloads.poisson(
        ctx.rng, rate=10.0, n=300, prompt_tokens=(16, 128), output_tokens=(4, 32), quantum_s=2**-10
    )
    giants = [
        workloads.SimRequest(r.id, r.arrival_s, 4000 if r.id % 9 == 0 else r.prompt_tokens,
                             r.output_tokens)
        for r in mixed
    ]  # fmt: skip
    uniforms = ctx.rng.random((len(giants), 2))
    check_routes(ctx, giants, uniforms, "small requests among very large ones", interval_s=3.0)
    check_routes(ctx, giants[:120], uniforms, "three replicas", interval_s=1.0, replicas=3)
