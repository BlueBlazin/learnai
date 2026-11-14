"""SimEngine, CostModel, SimRequest, and workloads.

All costs here are exact binary fractions, so virtual times can be compared with ==.
The last section builds a small continuous-batching scheduler on the engine, the way a
problem's reference would, to show that the pieces carry a realistic problem.
"""

import asyncio
from collections import deque

import numpy as np
import pytest

from whetstone.judge import SimClock, percentile
from whetstone.kit import (
    STEP_DONE_PRIORITY,
    CostModel,
    OutOfBlocks,
    ProtocolViolation,
    SimEngine,
    SimRequest,
    StepResult,
    workloads,
)

COST = CostModel(base_s=0.5, prefill_s_per_token=0.125, decode_s_per_seq=0.25)


def req(id, prompt, output, arrival=0.0, **more):
    return SimRequest(id, arrival, prompt, output, **more)


@pytest.fixture
def sim():
    with SimClock() as clock:
        yield clock


def events(sim, *kinds):
    """The trace as (time, kind, fields) tuples, optionally only some kinds."""
    return [(e.time, e.kind, e.fields) for e in sim.trace.events if not kinds or e.kind in kinds]


def violation(sim, engine, make_call, code, kind=ProtocolViolation):
    """Assert that ``make_call()`` (a call, or a coroutine to run) is refused with
    ``code`` and that the refusal changed nothing but the record of violations."""
    before = (sim.now(), engine.counters.snapshot(), engine.live, engine.busy, engine.free_blocks,
              len(sim.trace.events), len(engine.violations))  # fmt: skip
    with pytest.raises(ProtocolViolation) as caught:
        result = make_call()
        if asyncio.iscoroutine(result):
            sim.run_until_complete(result)
    assert caught.value.code == code and type(caught.value) is kind
    assert engine.violations[-1] is caught.value
    last = sim.trace.events[-1]
    assert (last.kind, last["code"], last["message"]) == ("violation", code, str(caught.value))
    after = (sim.now(), engine.counters.snapshot(), engine.live, engine.busy, engine.free_blocks,
             len(sim.trace.events) - 1, len(engine.violations) - 1)  # fmt: skip
    assert after == before
    return caught.value


# -- the cost model -----------------------------------------------------------------------


def test_cost_model():
    assert COST.step_s(0, 0) == 0.5
    assert COST.step_s(4, 0) == 1.0
    assert COST.step_s(0, 2) == 1.0
    assert COST.step_s(8, 3) == 0.5 + 1.0 + 0.75
    assert CostModel(0.0, 0.0, 0.0).step_s(100, 100) == 0.0
    with pytest.raises(ValueError, match="negative"):
        CostModel(0.1, -0.1, 0.1)
    with pytest.raises(AttributeError):
        COST.base_s = 1.0  # frozen
    with pytest.raises(TypeError):
        CostModel()  # no defaults: a problem states its costs


# -- the life of a sequence ---------------------------------------------------------------


def test_prefill_gives_the_first_token_and_each_decode_one_more(sim):
    engine = SimEngine(sim, COST)
    engine.admit(req("a", prompt=4, output=3))
    assert engine.live == ["a"] and engine.remaining_prefill("a") == 4 and not engine.busy

    async def run():
        results = [await engine.step(prefill=[("a", 4)])]
        assert engine.remaining_prefill("a") == 0 and engine.generated("a") == 1
        results.append(await engine.step(decode=["a"]))
        results.append(await engine.step(decode=["a"]))
        return results

    results = sim.run_until_complete(run())
    assert results == [
        StepResult(index=0, duration=1.0, tokens=("a",), finished=()),
        StepResult(index=1, duration=0.75, tokens=("a",), finished=()),
        StepResult(index=2, duration=0.75, tokens=("a",), finished=("a",)),
    ]
    assert sim.now() == 2.5 and engine.live == [] and not engine.busy and engine.busy_s == 2.5
    assert events(sim) == [
        (0.0, "admit", {"id": "a"}),
        (0.0, "step", {"index": 0, "prefill": (("a", 4),), "decode": (), "tokens": 4, "duration": 1.0}),
        (1.0, "first_token", {"id": "a", "step": 0}),
        (1.0, "token", {"id": "a", "index": 0, "step": 0}),
        (1.0, "step_done", {"index": 0}),
        (1.0, "step", {"index": 1, "prefill": (), "decode": ("a",), "tokens": 1, "duration": 0.75}),
        (1.75, "token", {"id": "a", "index": 1, "step": 1}),
        (1.75, "step_done", {"index": 1}),
        (1.75, "step", {"index": 2, "prefill": (), "decode": ("a",), "tokens": 1, "duration": 0.75}),
        (2.5, "token", {"id": "a", "index": 2, "step": 2}),
        (2.5, "finish", {"id": "a", "tokens": 3, "step": 2}),
        (2.5, "step_done", {"index": 2}),
    ]  # fmt: skip
    assert engine.counters.snapshot() == {
        "steps": 3, "prefill_tokens": 4, "decode_tokens": 2, "output_tokens": 3,
        "recomputed_tokens": 0, "preemptions": 0,
    }  # fmt: skip
    assert engine.violations == []


def test_a_single_output_token_finishes_with_the_prefill(sim):
    engine = SimEngine(sim, COST)
    engine.admit(req(1, prompt=2, output=1))
    result = sim.run_until_complete(engine.step(prefill=[(1, 2)]))
    assert result.tokens == (1,) and result.finished == (1,) and engine.live == []
    assert [e.kind for e in sim.trace.events] == [
        "admit", "step", "first_token", "token", "finish", "step_done",
    ]  # fmt: skip


def test_chunked_prefill_produces_the_token_only_when_the_prompt_is_complete(sim):
    engine = SimEngine(sim, COST)
    engine.admit(req("a", prompt=10, output=2))

    async def run():
        first = await engine.step(prefill=[("a", 4)])
        assert first.tokens == () and engine.remaining_prefill("a") == 6
        assert engine.generated("a") == 0 and engine.kv_tokens("a") == 4
        second = await engine.step(prefill=[("a", 5)])
        assert second.tokens == () and engine.remaining_prefill("a") == 1
        third = await engine.step(prefill=[("a", 1)])
        assert third.tokens == ("a",) and engine.kv_tokens("a") == 10
        return await engine.step(decode=["a"])

    assert sim.run_until_complete(run()).finished == ("a",)
    assert sim.trace.times("first_token") == [1.0 + 1.125 + 0.625]
    assert engine.counters.prefill_tokens == 10 and engine.counters.steps == 4


def test_a_mixed_step_reports_prefills_first_then_decodes(sim):
    engine = SimEngine(sim, COST)
    for request in (req("a", 4, 3), req("b", 2, 2), req("c", 8, 1)):
        engine.admit(request)

    async def run():
        await engine.step(prefill=[("a", 4)])
        mixed = await engine.step(decode=["a"], prefill=[("c", 4), ("b", 2)])
        assert mixed.duration == 0.5 + 6 * 0.125 + 0.25 and mixed.tokens == ("b", "a")
        last = await engine.step(prefill=[("c", 4)], decode=["b", "a"])
        assert last.tokens == ("c", "b", "a") and last.finished == ("c", "b", "a")

    sim.run_until_complete(run())
    assert sim.trace.values("token", "id") == ["a", "b", "a", "c", "b", "a"]
    assert sim.trace.values("first_token", "id") == ["a", "b", "c"]
    assert sim.trace.values("finish", "id") == ["c", "b", "a"]
    assert engine.live == [] and engine.violations == []


def test_the_engine_works_for_a_duck_typed_request_and_zero_cost(sim):
    class Row:
        id, prompt_tokens, output_tokens = (7, "x"), 3, 2

    engine = SimEngine(sim, CostModel(0.0, 0.0, 0.0))
    engine.admit(Row())

    async def run():
        await engine.step(prefill=[((7, "x"), 3)])
        return await engine.step(decode=[(7, "x")])

    assert sim.run_until_complete(run()).finished == ((7, "x"),)
    assert sim.now() == 0.0


# -- protocol violations ------------------------------------------------------------------


def test_admit_violations(sim):
    engine = SimEngine(sim, COST, max_seqs=2)
    engine.admit(req("a", 4, 2))
    violation(sim, engine, lambda: engine.admit(req("a", 9, 9)), "duplicate_sequence")

    class Bad:
        id, prompt_tokens, output_tokens = "z", 0, 2

    violation(sim, engine, lambda: engine.admit(Bad()), "bad_request")
    Bad.prompt_tokens, Bad.output_tokens = 3, True
    violation(sim, engine, lambda: engine.admit(Bad()), "bad_request")
    engine.admit(req("b", 4, 2))
    violation(sim, engine, lambda: engine.admit(req("c", 4, 2)), "too_many_sequences")
    assert engine.live == ["a", "b"] and len(engine.violations) == 4
    assert sim.trace.values("violation", "code") == [
        "duplicate_sequence", "bad_request", "bad_request", "too_many_sequences",
    ]  # fmt: skip
    # A finished sequence frees its place, and its id may be admitted again.
    sim.run_until_complete(engine.step(prefill=[("a", 4), ("b", 4)]))
    sim.run_until_complete(engine.step(decode=["a", "b"]))
    engine.admit(req("a", 1, 1))
    assert engine.live == ["a"]


def test_step_violations(sim):
    engine = SimEngine(sim, COST, max_batch_tokens=8)
    engine.admit(req("a", 6, 3))
    engine.admit(req("b", 6, 3))
    violation(sim, engine, lambda: engine.step(), "empty_step")
    violation(sim, engine, lambda: engine.step(prefill=[], decode=[]), "empty_step")
    violation(sim, engine, lambda: engine.step(prefill=[("a", 2), ("a", 2)]), "duplicate_sequence")
    violation(sim, engine, lambda: engine.step(prefill=[("a", 2)], decode=["a"]), "duplicate_sequence")
    violation(sim, engine, lambda: engine.step(prefill=[("nope", 2)]), "unknown_sequence")
    violation(sim, engine, lambda: engine.step(decode=["nope"]), "unknown_sequence")
    violation(sim, engine, lambda: engine.step(decode=[["unhashable"]]), "unknown_sequence")
    for n in (0, -1, 7, 2.0, True):
        violation(sim, engine, lambda n=n: engine.step(prefill=[("a", n)]), "bad_prefill")
    # Decoding before the prefill is complete: not started, and partly done.
    violation(sim, engine, lambda: engine.step(decode=["a"]), "prefill_incomplete")
    sim.run_until_complete(engine.step(prefill=[("a", 5)]))
    violation(sim, engine, lambda: engine.step(decode=["a"]), "prefill_incomplete")
    # In the same step as the chunk that completes it: still too early.
    violation(sim, engine, lambda: engine.step(prefill=[("a", 1)], decode=["a"]), "duplicate_sequence")
    sim.run_until_complete(engine.step(prefill=[("a", 1)]))
    violation(sim, engine, lambda: engine.step(prefill=[("a", 1)]), "bad_prefill")  # nothing remains
    # The budget counts prompt tokens plus decoded sequences: 6 + 2 fits 8, 7 + 2 does not.
    engine.admit(req("c", 7, 3))
    sim.run_until_complete(engine.step(prefill=[("b", 6)], decode=["a"]))
    violation(sim, engine, lambda: engine.step(prefill=[("c", 7)], decode=["a", "b"]), "budget_exceeded")
    sim.run_until_complete(engine.step(prefill=[("c", 6)], decode=["a", "b"]))
    # A finished sequence is no longer live.
    assert sim.trace.values("finish", "id") == ["a"]
    violation(sim, engine, lambda: engine.step(decode=["a"]), "unknown_sequence")
    violation(sim, engine, lambda: engine.release("a"), "unknown_sequence")
    violation(sim, engine, lambda: engine.preempt("a"), "unknown_sequence")
    assert all(type(v) is ProtocolViolation for v in engine.violations)


def test_the_first_failing_check_wins(sim):
    engine = SimEngine(sim, COST, max_batch_tokens=2, kv_blocks=1, block_tokens=1)
    engine.admit(req("a", 5, 2))
    engine.admit(req("b", 5, 2))
    # Everything is wrong with this step; the checks run in the documented order.
    everything = {"prefill": [("a", 9), ("a", 1), ("x", 1)], "decode": ["b", "y"]}
    violation(sim, engine, lambda: engine.step(**everything), "duplicate_sequence")
    violation(sim, engine, lambda: engine.step(prefill=[("x", 1), ("a", 9)], decode=["b"]),
              "unknown_sequence")  # fmt: skip
    violation(sim, engine, lambda: engine.step(prefill=[("a", 9), ("x", 1)], decode=["b"]),
              "bad_prefill")  # fmt: skip
    violation(sim, engine, lambda: engine.step(prefill=[("a", 5)], decode=["y", "b"]),
              "unknown_sequence")  # fmt: skip
    violation(sim, engine, lambda: engine.step(prefill=[("a", 5)], decode=["b"]),
              "prefill_incomplete")  # fmt: skip
    violation(sim, engine, lambda: engine.step(prefill=[("a", 5)]), "budget_exceeded")
    violation(sim, engine, lambda: engine.step(prefill=[("a", 2)]), "out_of_blocks", OutOfBlocks)


def test_one_step_at_a_time(sim):
    engine = SimEngine(sim, COST)
    engine.admit(req("a", 4, 2))
    engine.admit(req("b", 4, 2))

    async def run():
        first = asyncio.ensure_future(engine.step(prefill=[("a", 4)]))
        await asyncio.sleep(0.25)
        assert engine.busy
        with pytest.raises(ProtocolViolation) as caught:
            await engine.step(prefill=[("b", 4)])
        assert caught.value.code == "busy" and sim.now() == 0.25
        engine.admit(req("c", 1, 1))  # admitting during a step is fine
        assert (await first).tokens == ("a",)
        assert not engine.busy and sim.now() == 1.0
        await engine.step(prefill=[("b", 4)])  # and now it is free again

    sim.run_until_complete(run())
    assert [v.code for v in engine.violations] == ["busy"]
    assert sim.trace.values("step", "index") == [0, 1]


def test_a_caught_violation_is_still_on_record(sim):
    engine = SimEngine(sim, COST)

    async def sloppy():
        try:
            await engine.step(decode=["ghost"])
        except ProtocolViolation:
            pass

    sim.run_until_complete(sloppy())
    assert [v.code for v in engine.violations] == ["unknown_sequence"]
    assert sim.trace.values("violation", "code") == ["unknown_sequence"]
    assert isinstance(engine.violations[0], RuntimeError)


def test_asking_about_a_sequence_that_is_not_live_raises_without_a_record(sim):
    engine = SimEngine(sim, COST)
    for ask in (engine.remaining_prefill, engine.generated, engine.kv_tokens, engine.blocks):
        with pytest.raises(ProtocolViolation) as caught:
            ask("ghost")
        assert caught.value.code == "unknown_sequence"
    assert engine.violations == [] and sim.trace.events == []


# -- KV memory ----------------------------------------------------------------------------


def test_kv_blocks_are_taken_at_step_start_and_freed_at_finish(sim):
    engine = SimEngine(sim, COST, kv_blocks=4, block_tokens=4)
    assert engine.free_blocks == 4 and SimEngine(sim, COST).free_blocks is None
    engine.admit(req("a", 5, 5))
    assert engine.free_blocks == 4  # admission allocates nothing

    async def run():
        step = asyncio.ensure_future(engine.step(prefill=[("a", 5)]))
        await asyncio.sleep(0)
        assert (engine.kv_tokens("a"), engine.blocks("a"), engine.free_blocks) == (5, 2, 2)
        await step
        # The first token is not in the cache yet; each decode stores one more token.
        for kv, blocks in ((6, 2), (7, 2), (8, 2)):
            await engine.step(decode=["a"])
            assert (engine.kv_tokens("a"), engine.blocks("a")) == (kv, blocks)
        assert engine.generated("a") == 4 and engine.free_blocks == 2
        await engine.step(decode=["a"])  # the 9th token would need a third block: it fits
        assert engine.live == [] and engine.free_blocks == 4  # finished: everything is free

    sim.run_until_complete(run())


def test_out_of_blocks_is_raised_before_anything_happens(sim):
    engine = SimEngine(sim, COST, kv_blocks=3, block_tokens=4)
    engine.admit(req("a", 8, 9))
    engine.admit(req("b", 5, 9))
    sim.run_until_complete(engine.step(prefill=[("a", 8)]))
    assert engine.free_blocks == 1
    error = violation(sim, engine, lambda: engine.step(prefill=[("b", 5)]), "out_of_blocks", OutOfBlocks)
    assert isinstance(error, ProtocolViolation) and "4 KV blocks" in str(error)
    assert engine.kv_tokens("b") == 0 and engine.remaining_prefill("b") == 5  # no partial allocation
    sim.run_until_complete(engine.step(prefill=[("b", 4)]))  # exactly one block: fits
    assert engine.free_blocks == 0
    # "a" sits on a block boundary (8 tokens): its next decode needs a new block.
    violation(sim, engine, lambda: engine.step(decode=["a"]), "out_of_blocks", OutOfBlocks)
    engine.release("b")
    assert engine.free_blocks == 1
    sim.run_until_complete(engine.step(decode=["a"]))
    assert engine.kv_tokens("a") == 9 and engine.free_blocks == 0


def test_preempt_frees_memory_and_requires_recomputing_the_context(sim):
    engine = SimEngine(sim, COST, kv_blocks=8, block_tokens=4)
    engine.admit(req("a", 6, 4))

    async def run():
        await engine.step(prefill=[("a", 6)])
        await engine.step(decode=["a"])
        assert (engine.generated("a"), engine.kv_tokens("a"), engine.free_blocks) == (2, 7, 6)
        engine.preempt("a")
        assert engine.live == ["a"] and engine.free_blocks == 8 and engine.kv_tokens("a") == 0
        assert engine.generated("a") == 2  # produced tokens are kept
        assert engine.remaining_prefill("a") == 8  # prompt (6) plus produced tokens (2)
        with pytest.raises(ProtocolViolation) as caught:
            await engine.step(decode=["a"])
        assert caught.value.code == "prefill_incomplete"
        partial = await engine.step(prefill=[("a", 3)])
        assert partial.tokens == ()
        resumed = await engine.step(prefill=[("a", 5)])
        assert resumed.tokens == ("a",) and engine.generated("a") == 3
        assert engine.kv_tokens("a") == 8  # as if it had never been preempted: 6 + 3 - 1
        return await engine.step(decode=["a"])

    assert sim.run_until_complete(run()).finished == ("a",)
    # Token indices continue; nothing is produced twice, and there is one first token.
    assert sim.trace.values("token", "index") == [0, 1, 2, 3]
    assert sim.trace.values("first_token", "id") == ["a"]
    assert [(e["id"], e["generated"]) for e in sim.trace.of("preempt")] == [("a", 2)]
    assert engine.counters.snapshot() == {
        "steps": 5, "prefill_tokens": 14, "decode_tokens": 2, "output_tokens": 4,
        "recomputed_tokens": 8, "preemptions": 1,
    }  # fmt: skip


def test_preempting_before_the_first_token_restarts_the_prompt(sim):
    engine = SimEngine(sim, COST)
    engine.admit(req("a", 6, 2))
    sim.run_until_complete(engine.step(prefill=[("a", 4)]))
    engine.preempt("a")
    assert engine.remaining_prefill("a") == 6 and engine.generated("a") == 0
    assert sim.run_until_complete(engine.step(prefill=[("a", 6)])).tokens == ("a",)
    assert engine.counters.recomputed_tokens == 6


def test_release_and_preempt_during_a_step_forfeit_its_work(sim):
    engine = SimEngine(sim, COST, kv_blocks=10, block_tokens=4)
    for request in (req("a", 4, 5), req("b", 4, 5), req("c", 4, 5)):
        engine.admit(request)

    async def run():
        await engine.step(prefill=[("a", 4), ("b", 4), ("c", 4)])
        step = asyncio.ensure_future(engine.step(decode=["a", "b", "c"]))
        await asyncio.sleep(0.5)
        engine.release("a")  # canceled mid-step
        engine.preempt("b")
        assert engine.live == ["b", "c"] and engine.free_blocks == 8
        result = await step
        assert result.tokens == ("c",) and result.duration == 1.25  # the step is not shortened
        assert engine.generated("b") == 1 and engine.remaining_prefill("b") == 5
        engine.admit(req("a", 2, 1))  # the id is free again: a new sequence
        again = await engine.step(prefill=[("a", 2)])
        assert again.finished == ("a",)

    sim.run_until_complete(run())
    cancel_time = sim.trace.of("release")[0].time
    assert cancel_time == 2.5 and sim.trace.of("release")[0].fields == {"id": "a", "generated": 1}
    late = [e for e in sim.trace.of("token") if e["id"] == "a" and cancel_time < e.time < 3.5]
    assert late == []  # no token for the canceled request after its cancellation
    assert sim.trace.values("first_token", "id") == ["a", "b", "c", "a"]


def test_a_step_runs_to_its_end_even_if_the_caller_is_canceled(sim):
    engine = SimEngine(sim, COST)
    engine.admit(req("a", 4, 2))

    async def run():
        step = asyncio.ensure_future(engine.step(prefill=[("a", 4)]))
        await asyncio.sleep(0.5)
        step.cancel()
        await asyncio.sleep(0.25)
        assert engine.busy  # a forward pass cannot be taken back
        await asyncio.sleep(0.25)
        assert not engine.busy and engine.generated("a") == 1
        return await engine.step(decode=["a"])

    assert sim.run_until_complete(run()).finished == ("a",)
    assert sim.trace.times("step_done") == [1.0, 1.75]


def test_an_arrival_at_the_instant_a_step_ends_is_seen_before_the_scheduler_resumes(sim):
    engine = SimEngine(sim, COST)
    engine.admit(req("a", 4, 2))
    seen = []

    async def scheduler():
        await engine.step(prefill=[("a", 4)])  # ends at t = 1.0
        seen.append(("resumed", sim.now(), list(arrived)))

    arrived = []
    assert STEP_DONE_PRIORITY == 10
    sim.spawn(scheduler())
    # Scheduled after the step started, for the very instant the step ends.
    sim.call_at(0.5, lambda: sim.call_at(1.0, arrived.append, "late-scheduled"))
    sim.spawn_at(1.0, lambda: asyncio.sleep(0, arrived.append("spawned")))
    sim.run()
    assert seen == [("resumed", 1.0, ["spawned", "late-scheduled"])]
    assert sim.trace.times("token") == [1.0]


def test_block_tokens_must_be_positive(sim):
    with pytest.raises(ValueError):
        SimEngine(sim, COST, block_tokens=0)


# -- SimRequest and workloads -------------------------------------------------------------


def test_sim_request_fields_and_validation():
    request = SimRequest(3, 1.5, 10, 4)
    assert (request.priority, request.tenant, request.deadline_s) == (0, None, None)
    assert SimRequest(id="x", arrival_s=0.0, prompt_tokens=1, output_tokens=1, priority=2,
                      tenant="t", deadline_s=9.0).tenant == "t"  # fmt: skip
    with pytest.raises(AttributeError):
        request.priority = 1  # frozen
    for bad in ({"arrival_s": -1.0}, {"arrival_s": float("nan")}, {"prompt_tokens": 0},
                {"output_tokens": 0}, {"prompt_tokens": 2.0}, {"output_tokens": True}):  # fmt: skip
        with pytest.raises(ValueError):
            SimRequest(**{"id": 0, "arrival_s": 0.0, "prompt_tokens": 1, "output_tokens": 1, **bad})


def test_poisson_workload():
    requests = workloads.poisson(np.random.default_rng(5), rate=50.0, n=400,
                                 prompt_tokens=(8, 32), output_tokens=4,
                                 priorities=(0, 1, 2), tenants=("a", "b"), deadline_after_s=2.0)  # fmt: skip
    assert [r.id for r in requests] == list(range(400))
    arrivals = [r.arrival_s for r in requests]
    assert arrivals == sorted(arrivals) and arrivals[0] > 0
    assert 0.015 < np.mean(np.diff(arrivals)) < 0.025  # mean gap 1 / rate = 0.02
    assert {r.prompt_tokens for r in requests} == set(range(8, 33))  # both ends included
    assert {r.output_tokens for r in requests} == {4}
    assert {r.priority for r in requests} == {0, 1, 2} and {r.tenant for r in requests} == {"a", "b"}
    assert all(r.deadline_s == r.arrival_s + 2.0 for r in requests)
    assert all(type(r.prompt_tokens) is int and type(r.arrival_s) is float for r in requests)
    again = workloads.poisson(np.random.default_rng(5), rate=50.0, n=400,
                              prompt_tokens=(8, 32), output_tokens=4,
                              priorities=(0, 1, 2), tenants=("a", "b"), deadline_after_s=2.0)  # fmt: skip
    assert again == requests
    # Options do not shift the random numbers of the other columns.
    plain = workloads.poisson(np.random.default_rng(5), rate=50.0, n=400, prompt_tokens=(8, 32),
                              output_tokens=4)  # fmt: skip
    assert [r.arrival_s for r in plain] == arrivals
    assert [r.prompt_tokens for r in plain] == [r.prompt_tokens for r in requests]
    assert {(r.priority, r.tenant, r.deadline_s) for r in plain} == {(0, None, None)}
    shifted = workloads.poisson(np.random.default_rng(5), rate=50.0, n=3, start_s=10.0)
    assert shifted[0].arrival_s > 10.0
    assert workloads.poisson(np.random.default_rng(0), rate=1.0, n=0) == []
    with pytest.raises(ValueError):
        workloads.poisson(np.random.default_rng(0), rate=0.0, n=1)
    with pytest.raises(ValueError):
        workloads.poisson(np.random.default_rng(0), rate=1.0, n=1, prompt_tokens=(0, 4))


def test_quantized_arrivals_are_exact_and_can_coincide():
    quantum = 2**-4
    requests = workloads.poisson(np.random.default_rng(1), rate=40.0, n=200, quantum_s=quantum)
    arrivals = [r.arrival_s for r in requests]
    assert all((a / quantum).is_integer() for a in arrivals) and arrivals == sorted(arrivals)
    assert len(set(arrivals)) < len(arrivals)  # ties exist
    raw = workloads.poisson(np.random.default_rng(1), rate=40.0, n=200)
    assert all(0 <= r.arrival_s - q < quantum for r, q in zip(raw, arrivals, strict=True))


def test_bursty_workload():
    requests = workloads.bursty(np.random.default_rng(2), bursts=3, burst_size=4, period_s=2.0,
                                start_s=1.0, output_tokens=(1, 3))  # fmt: skip
    assert [r.arrival_s for r in requests] == [1.0] * 4 + [3.0] * 4 + [5.0] * 4  # exact ties
    assert [r.id for r in requests] == list(range(12))
    jittered = workloads.bursty(np.random.default_rng(2), bursts=3, burst_size=4, period_s=2.0,
                                jitter_s=0.5)  # fmt: skip
    arrivals = [r.arrival_s for r in jittered]
    assert arrivals == sorted(arrivals) and len(set(arrivals)) == 12
    assert all(2.0 * (i // 4) <= a < 2.0 * (i // 4) + 0.5 for i, a in enumerate(arrivals))
    assert workloads.bursty(np.random.default_rng(2), bursts=0, burst_size=4, period_s=1.0) == []
    with pytest.raises(ValueError):
        workloads.bursty(np.random.default_rng(2), bursts=1, burst_size=1, period_s=-1.0)


def test_replay_workload():
    requests = workloads.replay([
        (0.5, 10, 2),
        {"arrival_s": 0.0, "prompt_tokens": 3, "output_tokens": 1, "tenant": "t"},
        SimRequest("named", 0.5, 4, 4),
        (0.0, 7, 7, 2, "u", 9.0),
    ])  # fmt: skip
    assert [r.id for r in requests] == [1, 3, 0, "named"]  # sorted by time, ties in list order
    assert requests[1] == SimRequest(3, 0.0, 7, 7, priority=2, tenant="u", deadline_s=9.0)
    assert requests[0].tenant == "t"
    assert workloads.replay([]) == []
    assert workloads.replay([{"id": "a", "arrival_s": 1, "prompt_tokens": 1, "output_tokens": 1}])[0].id == "a"
    with pytest.raises(ValueError, match="unique"):
        workloads.replay([(0.0, 1, 1), SimRequest(0, 1.0, 1, 1)])
    with pytest.raises(ValueError):
        workloads.replay([(0.0, 0, 1)])


def test_drive_delivers_requests_on_time_in_order(sim):
    requests = workloads.replay([(1.0, 4, 1), (0.25, 4, 1), (1.0, 4, 1)])
    got = []

    async def handler(request):
        got.append(("start", request.id, sim.now()))
        await asyncio.sleep(0.5)
        got.append(("end", request.id, sim.now()))

    workloads.drive(sim, requests, lambda r: got.append(("sync", r.id, sim.now())))
    workloads.drive(sim, requests, handler)
    sim.run()
    assert [g for g in got if g[0] == "sync"] == [("sync", 1, 0.25), ("sync", 0, 1.0), ("sync", 2, 1.0)]
    assert [g for g in got if g[0] != "sync"] == [
        ("start", 1, 0.25), ("end", 1, 0.75), ("start", 0, 1.0), ("start", 2, 1.0),
        ("end", 0, 1.5), ("end", 2, 1.5),
    ]  # fmt: skip
    arrive = sim.trace.of("arrive")
    # One `arrive` event per delivery; at 1.0 the first drive's timers were scheduled first.
    assert [(e.time, e["id"]) for e in arrive] == [
        (0.25, 1), (0.25, 1), (1.0, 0), (1.0, 2), (1.0, 0), (1.0, 2),
    ]  # fmt: skip
    assert arrive[0]["request"] is requests[0]


# -- a reference scheduler built on the engine ---------------------------------------------


async def continuous_batching(engine, queue, max_running, total):
    """Iteration-level scheduling: admit into free slots before every step (first come,
    first served), prefill new prompts whole, decode everything that is running, and
    free a slot the moment its sequence finishes."""
    waiting, running, finished = deque(), [], 0
    while finished < total:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())  # idle: sleep until something arrives
        while not queue.empty():
            waiting.append(queue.get_nowait())
        prefill = []
        while waiting and len(running) + len(prefill) < max_running:
            request = waiting.popleft()
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))
        result = await engine.step(prefill=prefill, decode=running)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in result.finished]
        finished += len(result.finished)


async def static_batching(engine, queue, max_running, total):
    """The baseline: a batch is admitted only when the previous one has finished entirely."""
    waiting, running, finished = deque(), [], 0
    while finished < total:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())
        prefill = []
        while not running and waiting and len(prefill) < max_running:
            request = waiting.popleft()
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))
        result = await engine.step(prefill=prefill, decode=running)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in result.finished]
        finished += len(result.finished)


def simulate(scheduler, requests, *, max_running, cost=COST):
    """Run ``scheduler`` over ``requests``; return the (closed) simulation and the engine."""
    with SimClock() as sim:
        engine = SimEngine(sim, cost, max_seqs=max_running)
        queue = asyncio.Queue()
        workloads.drive(sim, requests, queue.put_nowait)
        sim.spawn(scheduler(engine, queue, max_running, len(requests)))
        sim.run()
        return sim, engine


def test_the_reference_scheduler_reproduces_a_hand_computed_trace():
    requests = workloads.replay([(0.0, 4, 3), (0.5, 2, 2), (2.0, 2, 1)])
    sim, engine = simulate(continuous_batching, requests, max_running=2)
    assert engine.violations == []
    assert events(sim, "step", "first_token", "token", "finish") == [
        (0.0, "step", {"index": 0, "prefill": ((0, 4),), "decode": (), "tokens": 4, "duration": 1.0}),
        (1.0, "first_token", {"id": 0, "step": 0}),
        (1.0, "token", {"id": 0, "index": 0, "step": 0}),
        # Request 1 arrived during step 0 and joins at once, next to the decode of 0.
        (1.0, "step", {"index": 1, "prefill": ((1, 2),), "decode": (0,), "tokens": 3, "duration": 1.0}),
        (2.0, "first_token", {"id": 1, "step": 1}),
        (2.0, "token", {"id": 1, "index": 0, "step": 1}),
        (2.0, "token", {"id": 0, "index": 1, "step": 1}),
        # Request 2 arrives at exactly 2.0, when step 1 ends, but both slots are taken.
        (2.0, "step", {"index": 2, "prefill": (), "decode": (0, 1), "tokens": 2, "duration": 1.0}),
        (3.0, "token", {"id": 0, "index": 2, "step": 2}),
        (3.0, "finish", {"id": 0, "tokens": 3, "step": 2}),
        (3.0, "token", {"id": 1, "index": 1, "step": 2}),
        (3.0, "finish", {"id": 1, "tokens": 2, "step": 2}),
        (3.0, "step", {"index": 3, "prefill": ((2, 2),), "decode": (), "tokens": 2, "duration": 0.75}),
        (3.75, "first_token", {"id": 2, "step": 3}),
        (3.75, "token", {"id": 2, "index": 0, "step": 3}),
        (3.75, "finish", {"id": 2, "tokens": 1, "step": 3}),
    ]  # fmt: skip
    assert sim.trace.latencies("arrive", "finish") == [3.0, 2.5, 1.75]
    assert sim.trace.latencies("arrive", "first_token") == [1.0, 1.5, 1.75]


def test_a_realistic_problem_can_be_judged_on_the_engine():
    cost = CostModel(base_s=2**-6, prefill_s_per_token=2**-12, decode_s_per_seq=2**-9)
    requests = workloads.poisson(np.random.default_rng(11), rate=12.0, n=150,
                                 prompt_tokens=(16, 128), output_tokens=(2, 40))  # fmt: skip
    by_id = {r.id: r for r in requests}
    results = {}
    for scheduler in (continuous_batching, static_batching):
        sim, engine = simulate(scheduler, requests, max_running=8, cost=cost)
        # Correctness: nothing illegal, every request served exactly, slots respected.
        assert engine.violations == [] and engine.live == [] and not engine.busy
        assert sorted(sim.trace.values("finish", "id")) == sorted(by_id)
        assert all(e["tokens"] == by_id[e["id"]].output_tokens for e in sim.trace.of("finish"))
        produced = {}
        for event in sim.trace.of("token"):
            assert event["index"] == produced.get(event["id"], 0)  # in order, none twice
            produced[event["id"]] = event["index"] + 1
        assert produced == {r.id: r.output_tokens for r in requests}
        assert engine.counters.output_tokens == sum(r.output_tokens for r in requests)
        assert engine.counters.prefill_tokens == sum(r.prompt_tokens for r in requests)
        assert max(len(e["prefill"]) + len(e["decode"]) for e in sim.trace.of("step")) <= 8
        # Steps never overlap, and no request is served before it arrives.
        starts, ends = sim.trace.times("step"), sim.trace.times("step_done")
        assert all(end <= nxt for end, nxt in zip(ends, starts[1:], strict=False))
        admitted = dict(zip(sim.trace.values("admit", "id"), sim.trace.times("admit"), strict=True))
        assert all(admitted[r.id] >= r.arrival_s for r in requests)
        assert sim.trace.values("admit", "id") == sorted(by_id)  # first come, first served
        results[scheduler.__name__] = (sim.trace.latencies("arrive", "finish"),
                                       sim.trace.latencies("arrive", "first_token"),
                                       events(sim))  # fmt: skip
    continuous, static = results["continuous_batching"], results["static_batching"]
    # The property a problem would grade: continuous batching cuts latency by a wide margin.
    assert percentile(continuous[0], 50) < 0.5 * percentile(static[0], 50)
    assert percentile(continuous[1], 99) < 0.5 * percentile(static[1], 99)
    # And the run is exactly reproducible, which is what allows comparing whole traces.
    sim, _ = simulate(continuous_batching, requests, max_running=8, cost=cost)
    assert events(sim) == continuous[2]
