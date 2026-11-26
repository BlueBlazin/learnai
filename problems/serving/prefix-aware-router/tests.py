"""Tests for prefix-aware-router.

`Router.route` is first called directly, with stand-in replicas that only have an
`id` and a `queue_len`, on scripts whose answers are written out. Then the router is
put in front of the kit's `SimCluster`, four replicas with a small prefix cache each,
on seeded workloads whose prompts share prefixes. There every decision must equal the
oracle's, and the two numbers the router exists for (prompt tokens served from a
cache, and the 95th-percentile time to first token) are compared with a router that
only looks at load.
"""

from types import SimpleNamespace

from learnai.judge import CheckFailure, case, check, percentile
from learnai.kit import CostModel, SimCluster, workloads

COST = CostModel(base_s=2**-6, prefill_s_per_token=2**-12, decode_s_per_seq=2**-9)
REPLICAS = 4
CACHE_TOKENS = 1200
THRESHOLD, IMBALANCE, INDEX_SIZE = 0.25, 4, 3


def fake(queues):
    """Stand-in replicas: {id: queue_len}, in the dict's order."""
    return [SimpleNamespace(id=i, queue_len=q) for i, q in queues.items()]


def play(ctx, config, script, label):
    """Run (tokens, {id: queue_len}, expected id, why) steps against one Router."""
    router = ctx.solution.Router(*config)
    history = [
        f"Router(match_threshold={config[0]}, imbalance={config[1]}, index_size={config[2]})"
    ]
    for tokens, queues, expected, why in script:
        got = router.route(tuple(tokens), fake(queues))
        history.append(f"route({tuple(tokens)!r}, queue_len={queues!r}) -> {got!r}")
        check.that(
            type(got) is int and got == expected,
            f"{label}: route returned {got!r}, expected replica {expected!r}",
            "\n".join(history) + f"\nexpected: {expected!r}  ({why})\n",
        )


def least_loaded(request, ready):
    """The load-only baseline: fewest requests, lowest id on ties."""
    return min(ready, key=lambda r: (r.queue_len, r.id))


def simulate(ctx, requests, make_route, *, replicas=REPLICAS, cache_tokens=CACHE_TOKENS):
    sim = ctx.simclock()
    route = make_route()
    routes, gaps = [], []

    def hook(request, ready):
        chosen = route(request, ready)
        chosen_id = chosen if isinstance(chosen, int) else getattr(chosen, "id", None)
        target = next((r for r in ready if r.id == chosen_id), None)
        if type(chosen_id) is not int or target is None:
            raise CheckFailure(f"route returned {chosen!r}, which is not the id of a replica")
        routes.append(chosen_id)
        gaps.append(target.queue_len - min(r.queue_len for r in ready))
        return chosen_id

    cluster = SimCluster(sim, COST, replicas=replicas, max_seqs=8, cache_tokens=cache_tokens,
                         block_tokens=16, route=hook)  # fmt: skip
    workloads.drive(sim, requests, cluster.submit)
    sim.run()
    check.that(not cluster.violations, "the router misused the cluster")
    ttft = sim.trace.latencies("arrive", "first_token")
    return SimpleNamespace(
        routes=routes, gap=max(gaps), hits=cluster.counters.cache_hit_tokens,
        p95=percentile(ttft, 95),
    )  # fmt: skip


def router_of(module, config=(THRESHOLD, IMBALANCE, INDEX_SIZE)):
    def make():
        router = module.Router(*config)
        return lambda request, ready: router.route(request.tokens, ready)

    return make


def check_routes(ctx, requests, label, config=(THRESHOLD, IMBALANCE, INDEX_SIZE), **cluster):
    want = simulate(ctx, requests, router_of(ctx.oracle, config), **cluster)
    got = simulate(ctx, requests, router_of(ctx.solution, config), **cluster)
    wrong = next(
        (i for i, (g, w) in enumerate(zip(got.routes, want.routes, strict=False)) if g != w), None
    )
    check.that(
        wrong is None and len(got.routes) == len(want.routes),
        f"{label}: {len(got.routes)} requests were routed, expected {len(want.routes)}"
        if wrong is None
        else f"{label}: request {wrong} was routed to replica {got.routes[wrong]}, expected "
        f"{want.routes[wrong]}",
        ""
        if wrong is None
        else (
            f"Router(match_threshold={config[0]}, imbalance={config[1]}, index_size={config[2]})\n"
            f"the first difference is request {wrong}: expected replica {want.routes[wrong]}, "
            f"got {got.routes[wrong]}\nroutes of the first 24 requests:\n"
            f"expected: {want.routes[:24]}\n     got: {got.routes[:24]}\n"
        ),
    )
    return got


def prefix_workload(ctx, *, rate, skew, n=600, groups=8):
    return workloads.shared_prefixes(
        ctx.rng, rate=rate, n=n, groups=groups, prefix_tokens=(256, 512), suffix_tokens=(8, 32),
        output_tokens=(4, 24), skew=skew,
    )  # fmt: skip


def check_against_baseline(ctx, requests, *, hit_ratio, p95_ratio):
    base = simulate(ctx, requests, lambda: least_loaded)
    got = simulate(ctx, requests, router_of(ctx.solution))
    check.that(len(got.routes) == len(requests), "not every request was routed")
    ctx.metric("baseline_cache_hit_tokens", base.hits, unit="tokens")
    ctx.metric("baseline_p95_ttft", base.p95 * 1000, unit="ms")
    check.at_most("largest_queue_gap", got.gap, IMBALANCE, unit="requests")
    need = int(hit_ratio * base.hits)
    ctx.metric("cache_hit_tokens", got.hits, unit="tokens", limit=need, higher_is_better=True)
    check.that(
        got.hits >= need,
        f"cache_hit_tokens: {got.hits} tokens is below the required {need} tokens "
        f"({hit_ratio:g} times the load-only router's)",
    )
    check.at_most("p95_ttft", got.p95 * 1000, p95_ratio * base.p95 * 1000, unit="ms")


A = (1, 2, 3, 4, 5, 6, 7, 8)


@case("the worked example: follow a prefix, give it up when the replica is too busy",
      public=True)  # fmt: skip
def worked_example(ctx):
    play(ctx, (0.5, 2, 2), [
        (A, {0: 0, 1: 0, 2: 0}, 0, "nothing is indexed: least loaded, lowest id"),
        ((1, 2, 3, 4, 5, 6, 9, 9), {0: 1, 1: 0, 2: 0}, 0,
         "replica 0 shares 6 of 8 tokens and is within 2 of the least loaded"),
        ((1, 2, 3, 7, 7, 7, 7, 7), {0: 2, 1: 0, 2: 0}, 1,
         "the best match is 3 of 8 tokens, below 0.5: least loaded, lowest id"),
        ((1, 2, 3, 4, 5, 6, 7, 7), {0: 3, 1: 1, 2: 0}, 2,
         ("replica 0 is 3 above the least loaded, so it is not eligible; replica 1 shares "
         "only 3 tokens: least loaded")),
        ((1, 2, 3, 4, 5, 6, 7, 0), {0: 2, 1: 1, 2: 1}, 2,
         "replicas 0 and 2 both share 7 tokens; replica 2 has the shorter queue"),
    ], "the worked example")  # fmt: skip


@case("a prefix is the leading tokens up to the first difference, not tokens in common",
      public=True)  # fmt: skip
def prefix_means_prefix(ctx):
    # Rejects counts_equal_positions and exact_match_only.
    play(ctx, (0.5, 8, 4), [
        ((1, 2, 3, 4, 5, 6, 7, 8), {0: 0, 1: 0}, 0, "nothing indexed"),
        ((9, 9, 9, 9, 9, 9, 9, 9), {0: 1, 1: 0}, 1, "nothing in common: least loaded"),
        ((0, 2, 3, 4, 5, 6, 7, 8), {0: 0, 1: 2}, 0,
         ("the first token differs from everything indexed: the match is 0 on both replicas, "
         "so least loaded")),
        ((9, 9, 9, 9, 9, 0, 9, 9), {0: 0, 1: 2}, 1,
         "replica 1 shares the first 5 of 8 tokens"),
        ((1, 2, 3, 4), {0: 3, 1: 0}, 0,
         "the whole prompt is a prefix of one on replica 0: 4 of 4 tokens"),
        ((1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 1, 1, 1, 1), {0: 3, 1: 0}, 0,
         "8 of 16 tokens is exactly the threshold 0.5: it counts"),
        ((1, 2, 3, 4, 5, 6, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1), {0: 3, 1: 0}, 1,
         "7 of 16 tokens is below 0.5: least loaded"),
    ], "prefix matching")  # fmt: skip


@case("the imbalance bound is relative to the least loaded replica; ties go to the shorter "
      "queue, then the lower id", public=True)  # fmt: skip
def bound_and_ties(ctx):
    # Rejects affinity_never_broken, bound_is_absolute, first_best_match.
    play(ctx, (0.5, 2, 4), [
        (A, {5: 7, 3: 9, 8: 7}, 5, "nothing indexed: least loaded is a tie, lowest id"),
        (A, {5: 9, 3: 9, 8: 7}, 5, "replica 5 matches and is exactly 2 above the least loaded"),
        (A, {5: 10, 3: 9, 8: 7}, 8, "replica 5 is 3 above the least loaded: not eligible"),
    ], "the bound")  # fmt: skip
    play(ctx, (0.5, 2, 4), [
        (A, {8: 0, 5: 1}, 8, "nothing indexed"),
        (A, {8: 5, 5: 0}, 5, "replica 8 is not eligible: least loaded"),
        (A, {8: 4, 5: 4, 2: 3}, 5, "8 and 5 match equally with equal queues: the lower id"),
        (A, {8: 4, 5: 5, 2: 3}, 8, "8 and 5 match equally: the shorter queue"),
        (A, {8: 40, 5: 41, 2: 39}, 8, ("queues far above the bound in absolute terms are "
         "still within 2 of the least loaded")),
    ], "ties among equal matches")  # fmt: skip


@case("uniform prefixes on four replicas: cache hits and p95 TTFT against a load-only router",
      public=True)  # fmt: skip
def uniform_prefixes(ctx):
    # Rejects least_loaded_only and index_not_updated (no cache gain).
    requests = prefix_workload(ctx, rate=30.0, skew=0.0)
    check_against_baseline(ctx, requests, hit_ratio=1.8, p95_ratio=0.6)


@case("a hot prefix: affinity must give way before one replica is buried", public=True)
def hot_prefix(ctx):
    # Rejects affinity_never_broken: one replica takes most of the traffic.
    requests = prefix_workload(ctx, rate=40.0, skew=2.0)
    check_against_baseline(ctx, requests, hit_ratio=1.1, p95_ratio=0.75)


@case("the index holds the last index_size prompts routed to each replica, no more")
def index_is_bounded(ctx):
    # Rejects index_unbounded and index_keeps_first.
    b, c, d = (20, 21, 22, 23), (30, 31, 32, 33), (40, 41, 42, 43)
    play(ctx, (0.5, 9, 2), [
        (A, {0: 0, 1: 0}, 0, "nothing indexed"),
        (b, {0: 0, 1: 5}, 0, "least loaded; replica 0 now holds A and b"),
        (c, {0: 0, 1: 5}, 0, "least loaded; replica 0 now holds b and c: A is forgotten"),
        (A, {0: 6, 1: 0}, 1, "A is no longer indexed on replica 0: least loaded"),
        (b, {0: 6, 1: 0}, 0, "b is still indexed on replica 0"),
        (d, {0: 0, 1: 5}, 0, "least loaded; replica 0 now holds b (routed again) and d"),
        (c, {0: 6, 1: 0}, 1, "c was pushed out by the second b and d"),
        (A, {0: 0, 1: 6}, 1, "A is indexed on replica 1"),
    ], "index_size = 2")  # fmt: skip
    play(ctx, (0.5, 9, 1), [
        (A, {0: 0, 1: 0}, 0, "nothing indexed"),
        (A, {0: 3, 1: 0}, 0, "A is indexed on replica 0"),
        (b, {0: 0, 1: 3}, 0, "least loaded; b replaces A"),
        (A, {0: 3, 1: 0}, 1, "replica 0 only remembers b"),
    ], "index_size = 1")  # fmt: skip


@case("thresholds 0 and 1, an empty index for a new replica, replicas that disappear")
def edge_configurations(ctx):
    # Rejects threshold_strict (a match rate equal to the threshold) and zero-match affinity.
    play(ctx, (1.0, 9, 4), [
        (A, {0: 0, 1: 0}, 0, "nothing indexed"),
        (A, {0: 5, 1: 0}, 0, "an identical prompt has match rate 1.0, which reaches 1.0"),
        ((*A, 9), {0: 5, 1: 0}, 1, "8 of 9 tokens is below 1.0"),
    ], "match_threshold = 1")  # fmt: skip
    play(ctx, (0.0, 1, 4), [
        (A, {0: 3, 1: 2, 2: 2}, 1, ("nothing matches: among eligible replicas the best match "
         "is 0 everywhere, the shortest queue wins, then the lower id")),
        ((9, 9), {0: 2, 1: 3, 2: 3}, 0, ("threshold 0 accepts a match of 0: the eligible "
         "replica with the shortest queue")),
        (A, {0: 9, 2: 9, 7: 9}, 0, ("replica 1 is gone; 0 and 2 never saw A, 7 is new: "
         "all match 0, equal queues, the lower id")),
        (A, {2: 9, 7: 9}, 2, "replica 0, which holds A now, is not offered"),
        (A, {0: 9, 2: 9, 7: 8}, 0, "replicas 0 and 2 hold A; equal queues, lower id"),
    ], "match_threshold = 0")  # fmt: skip


@case("four replicas, uniform prefixes: every routing decision equals the specified one")
def exact_routes_uniform(ctx):
    requests = prefix_workload(ctx, rate=30.0, skew=0.0, n=400)
    check_routes(ctx, requests, "threshold 0.25, imbalance 4, index 3")
    check_routes(ctx, requests, "threshold 0.9, imbalance 1, index 8", (0.9, 1, 8))
    check_routes(ctx, requests, "threshold 0.25, imbalance 0, index 2", (0.25, 0, 2))


@case("skewed prefixes, bursts, and a short index: decisions equal the specified ones")
def exact_routes_skewed(ctx):
    requests = prefix_workload(ctx, rate=40.0, skew=1.5, n=400, groups=12)
    check_routes(ctx, requests, "twelve prefixes, skew 1.5")
    check_routes(ctx, requests, "three replicas", (0.25, 2, 1), replicas=3)
    burst = workloads.shared_prefixes(
        ctx.rng, rate=60.0, n=300, groups=5, prefix_tokens=(64, 200), suffix_tokens=(4, 64),
        output_tokens=(2, 12), skew=1.0, quantum_s=2**-4,
    )  # fmt: skip
    check_routes(ctx, burst, "arrivals in bursts", (0.5, 3, 4), replicas=5, cache_tokens=600)
