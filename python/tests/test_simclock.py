import asyncio
import time

import pytest

from whetstone.judge import CheckFailure, SimClock, percentile
from whetstone.kit import SimModel


def test_virtual_time_jumps_without_real_sleeping():
    with SimClock() as sim:
        seen = []
        sim.call_at(3600.0, lambda: seen.append(sim.now()))
        sim.call_later(0.25, lambda: seen.append(sim.now()))
        started = time.perf_counter()
        sim.run()
        assert time.perf_counter() - started < 0.5
        assert seen == [0.25, 3600.0]
        assert sim.now() == 3600.0


def test_callbacks_run_in_time_order_and_can_be_canceled():
    with SimClock() as sim:
        order = []
        sim.call_at(0.3, order.append, "c")
        sim.call_at(0.1, order.append, "a")
        handle = sim.call_at(0.2, order.append, "never")
        sim.call_at(0.2, order.append, "b")
        handle.cancel()
        sim.run()
        assert order == ["a", "b", "c"]


def test_same_instant_callbacks_run_in_schedule_order():
    # asyncio alone orders timers by time only; equal times came out in heap order.
    with SimClock() as sim:
        order = []
        for i in range(50):
            sim.call_at(1.0, order.append, i)
        sim.call_at(0.5, order.append, "early")
        sim.run()
        assert order == ["early", *range(50)]


def test_schedule_order_holds_across_every_way_of_scheduling():
    with SimClock() as sim:
        order = []

        async def sleeper(name):
            await asyncio.sleep(2.0)
            order.append(name)

        async def spawned(name):
            order.append(name)

        sim.call_at(2.0, order.append, "call_at")
        sim.spawn(sleeper("sleep"))  # its timer is created when the task first runs
        sim.call_later(2.0, order.append, "call_later")
        sim.loop.call_at(2.0, order.append, "loop.call_at")
        sim.spawn_at(2.0, lambda: spawned("spawn_at"))
        sim.loop.call_later(2.0, order.append, "loop.call_later")
        sim.run()
        # Timers fire in schedule order. The two coroutines run after the plain
        # callbacks of the instant: the sleeper is woken, and the spawned task is
        # created, by their timers, and both then wait their turn in the ready queue.
        assert order == [
            "call_at", "call_later", "loop.call_at", "loop.call_later", "spawn_at", "sleep",
        ]  # fmt: skip
        assert sim.now() == 2.0


def test_priority_orders_callbacks_of_one_instant():
    with SimClock() as sim:
        order = []
        sim.call_at(1.0, order.append, "default-1")
        sim.call_at(1.0, order.append, "late", priority=5)
        sim.call_at(1.0, order.append, "first", priority=-1)
        sim.call_later(1.0, order.append, "default-2")
        sim.call_at(1.0, order.append, "first-too", priority=-1)
        sim.call_at(0.5, order.append, "earlier time wins over priority", priority=99)
        sim.run()
        assert order == [
            "earlier time wins over priority",
            "first", "first-too", "default-1", "default-2", "late",
        ]  # fmt: skip
        with pytest.raises(TypeError):
            sim.call_at(2.0, order.append, "x", priority=0.5)


def test_spawn_at_starts_tasks_in_order_and_honors_priority():
    with SimClock() as sim:
        started = []

        async def client(i):
            started.append(i)

        for i in range(5):
            sim.spawn_at(1.0, lambda i=i: client(i))
        sim.spawn_at(1.0, lambda: client("urgent"), priority=-1)
        sim.run()
        assert started == ["urgent", 0, 1, 2, 3, 4]


def test_a_timer_scheduled_during_its_own_instant_runs_after_what_is_ready():
    with SimClock() as sim:
        order = []

        def first():
            order.append("first")
            sim.loop.call_soon(order.append, "soon")
            sim.call_later(0, order.append, "zero-delay", priority=-10)

        sim.call_at(1.0, first)
        sim.call_at(1.0, order.append, "second")
        sim.run()
        assert order == ["first", "second", "soon", "zero-delay"]
        assert sim.now() == 1.0


def test_same_instant_order_is_identical_on_every_run():
    def simulate():
        with SimClock() as sim:
            log = []

            async def worker(i):
                await sim.sleep(0.25 * (i % 3))
                log.append(("woke", i, sim.now()))
                await sim.sleep(0.5)
                log.append(("done", i, sim.now()))

            for i in range(30):
                sim.spawn_at(0.25 * (i % 4), lambda i=i: worker(i))
            sim.run()
            return log

    first = simulate()
    assert all(simulate() == first for _ in range(5))
    # Within one instant, workers wake in the order their sleeps were started.
    at_half = [i for kind, i, when in first if kind == "woke" and when == 0.5]
    assert at_half == sorted(at_half, key=lambda i: (-(i % 3), i))


def test_canceled_timers_do_not_disturb_the_order():
    with SimClock() as sim:
        order = []
        handles = [sim.call_at(1.0, order.append, i) for i in range(200)]
        for handle in handles[::2]:
            handle.cancel()  # enough to make asyncio rebuild its timer heap
        sim.run()
        assert order == list(range(1, 200, 2))


def test_plain_asyncio_code_runs_on_virtual_time():
    async def worker(queue, log):
        while True:
            item = await queue.get()
            await asyncio.sleep(1.5)
            log.append((asyncio.get_running_loop().time(), item))

    with SimClock() as sim:
        queue, log = asyncio.Queue(), []
        sim.spawn(worker(queue, log))
        sim.call_at(10.0, queue.put_nowait, "x")
        sim.call_at(10.5, queue.put_nowait, "y")
        sim.run()  # returns although the worker is still blocked on the queue
        assert log == [(11.5, "x"), (13.0, "y")]

        async def slow():
            await asyncio.sleep(100)

        with pytest.raises(asyncio.TimeoutError):
            sim.run_until_complete(asyncio.wait_for(slow(), timeout=5))
        assert sim.now() == 18.0


def test_run_until_stops_at_the_given_time_and_can_resume():
    with SimClock() as sim:
        ticks = []

        async def ticker():
            while True:
                await sim.sleep(1.0)
                ticks.append(sim.now())

        sim.spawn(ticker())
        sim.run(until=3.5)
        assert ticks == [1.0, 2.0, 3.0]
        assert sim.now() == 3.5
        sim.run(until=5.0)
        assert ticks == [1.0, 2.0, 3.0, 4.0, 5.0]
        with pytest.raises(ValueError):
            sim.run(until=1.0)


def test_run_until_advances_time_even_when_idle():
    with SimClock() as sim:
        sim.run(until=7.0)
        assert sim.now() == 7.0


def test_run_until_complete_returns_results_and_detects_deadlock():
    with SimClock() as sim:
        assert sim.run_until_complete(asyncio.sleep(2.0, result="done")) == "done"
        assert sim.now() == 2.0
        with pytest.raises(CheckFailure, match="went idle before the awaited operation finished"):
            sim.run_until_complete(asyncio.Event().wait())


def test_exceptions_in_tasks_and_callbacks_fail_the_run():
    async def boom():
        await asyncio.sleep(1)
        raise ValueError("from a task")

    with SimClock() as sim:
        sim.spawn(boom())
        with pytest.raises(ValueError, match="from a task"):
            sim.run()

    def bad_callback():
        raise KeyError("from a callback")

    with SimClock() as sim:
        sim.call_later(1, bad_callback)
        with pytest.raises(KeyError):
            sim.run()


def test_simulation_is_deterministic():
    def simulate():
        with SimClock() as sim:

            async def client(i):
                await sim.sleep(0.1 * (i % 3))
                sim.trace.record("arrive", id=i)
                await sim.sleep(0.05 * i)
                sim.trace.record("done", id=i)

            for i in range(20):
                sim.spawn(client(i))
            sim.run()
            return [(e.time, e.kind, e["id"]) for e in sim.trace.events]

    assert simulate() == simulate()


def test_trace_queries_and_latencies():
    with SimClock() as sim:
        sim.call_at(1.0, lambda: sim.trace.record("arrive", id="a"))
        sim.call_at(2.0, lambda: sim.trace.record("arrive", id="b"))
        sim.call_at(2.5, lambda: sim.trace.record("done", id="a", size=3))
        sim.call_at(4.0, lambda: sim.trace.record("done", id="b", size=5))
        sim.run()
        assert sim.trace.times("arrive") == [1.0, 2.0]
        assert sim.trace.values("done", "size") == [3, 5]
        assert [e["id"] for e in sim.trace.of("done")] == ["a", "b"]
        assert sim.trace.latencies("arrive", "done") == [1.5, 2.0]
        sim.trace.record("arrive", id="lost")
        with pytest.raises(CheckFailure, match="arrive 'lost' never reached 'done'"):
            sim.trace.latencies("arrive", "done")


def test_percentile_is_nearest_rank():
    values = [4, 1, 3, 2]
    assert percentile(values, 50) == 2
    assert percentile(values, 75) == 3
    assert percentile(values, 100) == 4
    assert percentile(values, 0) == 1
    assert percentile(range(1, 101), 99) == 99
    with pytest.raises(ValueError):
        percentile([], 50)


def test_sim_model_takes_virtual_time_and_enforces_its_rules():
    with SimClock() as sim:
        model = SimModel(sim, base_s=0.010, per_item_s=0.002, max_batch_size=4, fn=str.upper)
        assert sim.run_until_complete(model(["a", "b", "c"])) == ["A", "B", "C"]
        assert sim.now() == pytest.approx(0.016)
        assert model.counters.snapshot() == {"batches": 1, "items": 3}
        assert sim.trace.values("batch", "size") == [3]
        assert sim.trace.times("batch_done") == [pytest.approx(0.016)]
        with pytest.raises(RuntimeError, match="exceeds the model's max_batch_size"):
            sim.run_until_complete(model(list("abcde")))
        with pytest.raises(RuntimeError, match="empty batch"):
            sim.run_until_complete(model([]))

        async def overlap():
            await asyncio.gather(model(["a"]), model(["b"]))

        with pytest.raises(RuntimeError, match="one batch at a time"):
            sim.run_until_complete(overlap())


def test_close_cancels_pending_tasks_and_is_idempotent():
    sim = SimClock()
    cancelled = []

    async def forever():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    sim.spawn(forever())
    sim.run()
    sim.close()
    sim.close()
    assert cancelled == [True]
    with pytest.raises(RuntimeError):
        sim.run()
