from learnai.judge import case, check, percentile
from learnai.kit import SimModel


def simulate(ctx, batcher_class, arrivals, *, max_batch_size, max_wait_s):
    """Drive a Batcher with requests arriving at the given virtual times."""
    sim = ctx.simclock()
    model = SimModel(
        sim,
        base_s=0.004,
        per_item_s=0.0005,
        max_batch_size=max_batch_size,
        fn=lambda item: item * 2,
    )
    batcher = batcher_class(model, max_batch_size=max_batch_size, max_wait_s=max_wait_s)

    async def client(i):
        sim.trace.record("arrive", id=i)
        result = await batcher.submit(i)
        sim.trace.record("done", id=i, result=result)

    for i, when in enumerate(arrivals):
        sim.spawn_at(when, lambda i=i: client(i))
    sim.run()
    return sim


@case("a burst that fills a batch is dispatched immediately", public=True)
def full_batch(ctx):
    sim = simulate(ctx, ctx.solution.Batcher, [0.0] * 4, max_batch_size=4, max_wait_s=0.050)
    check.equal(sim.trace.values("batch", "size"), [4], label="batch sizes")
    check.equal(sim.trace.times("batch"), [0.0], label="dispatch times")


@case("a lone request is dispatched after max_wait_s", public=True)
def lone_request(ctx):
    sim = simulate(ctx, ctx.solution.Batcher, [0.0], max_batch_size=4, max_wait_s=0.050)
    check.equal(sim.trace.values("batch", "size"), [1], label="batch sizes")
    check.allclose(sim.trace.times("batch"), [0.050], label="dispatch times")
    check.equal(sim.trace.values("done", "result"), [0], label="results")


@case("every request gets its own result, and tail latency is bounded")
def steady_load(ctx):
    arrivals = [0.002 * i for i in range(100)]
    sim = simulate(ctx, ctx.solution.Batcher, arrivals, max_batch_size=8, max_wait_s=0.005)
    latencies = sim.trace.latencies("arrive", "done")
    for event in sim.trace.of("done"):
        check.that(
            event["result"] == 2 * event["id"], "a request received another request's result"
        )
    check.that(max(sim.trace.values("batch", "size")) <= 8, "a batch exceeded max_batch_size")
    check.at_most("p99_latency", percentile(latencies, 99) * 1000, 15.0, unit="ms")


@case("the same batches as the reference under bursty load")
def same_schedule(ctx):
    arrivals = sorted(float(t) for t in ctx.rng.uniform(0, 0.3, size=60))
    options = {"max_batch_size": 6, "max_wait_s": 0.010}
    got = simulate(ctx, ctx.solution.Batcher, arrivals, **options)
    want = simulate(ctx, ctx.oracle.Batcher, arrivals, **options)
    check.equal(
        got.trace.values("batch", "size"), want.trace.values("batch", "size"), label="batch sizes"
    )
    check.allclose(got.trace.times("batch"), want.trace.times("batch"), label="dispatch times")
