"""ScriptedLM: the scripted token source for service problems."""

import asyncio
import json
import threading

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from learnai.judge import SimClock, iter_sse, serve
from learnai.kit import ScriptedLM, ScriptedStream, StreamRecord

SCRIPT = ["Hello", ",", " world", "!"]


async def collect(stream):
    return [piece async for piece in stream]


def test_streams_the_script_and_records_the_call():
    lm = ScriptedLM(SCRIPT)
    assert not lm.closed and lm.active == 0 and lm.calls == []
    stream = lm.stream("say hi")
    assert isinstance(stream, ScriptedStream) and lm.active == 1 and not lm.closed
    assert asyncio.run(collect(stream)) == SCRIPT
    (record,) = lm.calls
    assert isinstance(record, StreamRecord) and stream.record is record
    assert (record.prompt, record.max_tokens, record.script) == ("say hi", None, tuple(SCRIPT))
    assert record.pieces == SCRIPT and record.closed and record.finish_reason == "stop"
    assert lm.closed and lm.active == 0 and not lm.stalled
    assert lm.counters.snapshot() == {"streams": 1, "pieces": 4}


@pytest.mark.parametrize(
    ("max_tokens", "pieces", "reason"),
    [(None, 4, "stop"), (10, 4, "stop"), (5, 4, "stop"), (4, 4, "length"), (2, 2, "length"),
     (1, 1, "length"), (0, 0, "length")],
)  # fmt: skip
def test_max_tokens_and_finish_reason(max_tokens, pieces, reason):
    lm = ScriptedLM(SCRIPT)
    assert asyncio.run(collect(lm.stream("p", max_tokens))) == SCRIPT[:pieces]
    assert lm.calls[0].finish_reason == reason and lm.calls[0].limit == pieces and lm.closed


@pytest.mark.parametrize("bad", [-1, 1.5, "3", True])
def test_max_tokens_is_validated(bad):
    lm = ScriptedLM(SCRIPT)
    with pytest.raises(ValueError, match="max_tokens"):
        lm.stream("p", bad)
    assert lm.calls == []


def test_script_forms():
    by_prompt = ScriptedLM({"a": ["1"], "b": ["2", "3"], None: ["?"]})
    assert asyncio.run(collect(by_prompt.stream("b"))) == ["2", "3"]
    assert asyncio.run(collect(by_prompt.stream("zzz"))) == ["?"]
    with pytest.raises(KeyError):
        ScriptedLM({"a": ["1"]}).stream("b")
    echo = ScriptedLM(lambda prompt: prompt.split())
    assert asyncio.run(collect(echo.stream("x y z", 2))) == ["x", "y"]
    assert asyncio.run(collect(ScriptedLM([]).stream("p"))) == []
    assert [r.prompt for r in by_prompt.calls] == ["b", "zzz"]


def test_count_tokens_is_words():
    lm = ScriptedLM(SCRIPT)
    assert lm.count_tokens("the quick  brown\nfox") == 4
    assert lm.count_tokens("") == 0 and lm.count_tokens("   ") == 0


def test_closed_tracks_exhaustion_aclose_and_abandonment():
    async def run():
        lm = ScriptedLM(SCRIPT)
        abandoned = lm.stream("a")
        assert await anext(abandoned) == "Hello"
        assert not lm.closed and lm.active == 1  # read from, then left open
        closed = lm.stream("b")
        assert await anext(closed) == "Hello"
        await closed.aclose()
        await closed.aclose()  # twice is fine
        assert closed.record.closed and closed.record.finish_reason is None
        with pytest.raises(StopAsyncIteration):
            await anext(closed)  # a closed stream yields nothing more
        assert closed.record.pieces == ["Hello"]
        assert not lm.closed and lm.active == 1  # the first one is still open
        await abandoned.aclose()
        assert lm.closed
        never_read = lm.stream("c")
        assert not lm.closed and never_read.record.pieces == []
        await never_read.aclose()
        return lm

    lm = asyncio.run(run())
    assert lm.counters.snapshot() == {"streams": 3, "pieces": 2} and lm.wait_closed(0.0)


def test_cancellation_while_producing_closes_the_stream():
    async def run():
        lm = ScriptedLM(SCRIPT, delay_s=10.0)
        task = asyncio.ensure_future(collect(lm.stream("p")))
        await asyncio.sleep(0.01)
        assert lm.active == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return lm

    lm = asyncio.run(run())
    assert lm.closed and lm.calls[0].pieces == [] and lm.calls[0].finish_reason is None


def test_delay_is_virtual_under_a_simulated_clock():
    with SimClock() as sim:
        lm = ScriptedLM(SCRIPT, delay_s=0.25)
        times = []

        async def consume():
            async for _ in lm.stream("p"):
                times.append(sim.now())

        sim.run_until_complete(consume())
        assert times == [0.25, 0.5, 0.75, 1.0] and lm.closed


def test_the_gate_holds_pieces_until_released():
    async def run():
        lm = ScriptedLM(SCRIPT, gated=True)
        got = []

        async def consume():
            async for piece in lm.stream("p"):
                got.append(piece)

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.02)
        assert got == []  # nothing until the test says so
        lm.release(1)
        await asyncio.sleep(0.02)
        assert got == ["Hello"]
        lm.release(2)
        await asyncio.sleep(0.02)
        assert got == ["Hello", ",", " world"] and lm.active == 1
        lm.release()  # open for good
        await task
        assert got == SCRIPT and lm.closed and not lm.stalled
        assert await collect(lm.stream("again")) == SCRIPT  # the gate stays open
        lm.release(5)  # no effect once open

    asyncio.run(run())


def test_a_gate_that_is_never_released_stalls_instead_of_hanging():
    lm = ScriptedLM(SCRIPT, gated=True, gate_timeout_s=0.05)
    lm.release(1)
    assert asyncio.run(collect(lm.stream("p"))) == ["Hello"]
    assert lm.stalled and lm.closed and lm.calls[0].finish_reason is None
    with SimClock() as sim:  # the timeout is in the loop's time: virtual here
        slow = ScriptedLM(SCRIPT, gated=True, gate_timeout_s=30.0)
        assert sim.run_until_complete(collect(slow.stream("p"))) == []
        assert slow.stalled and 30.0 <= sim.now() < 30.1


def make_app(lm):
    async def generate(request):
        body = await request.json()

        async def events():
            stream = lm.stream(body["prompt"], body.get("max_tokens"))
            try:
                async for piece in stream:
                    yield f"data: {json.dumps({'piece': piece})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                await stream.aclose()

        return StreamingResponse(events(), media_type="text/event-stream")

    async def status(request):
        return JSONResponse({"active": lm.active})

    return Starlette(
        routes=[Route("/generate", generate, methods=["POST"]), Route("/status", status)]
    )


def test_a_real_server_streams_piece_by_piece_under_the_gate():
    lm = ScriptedLM(SCRIPT, gated=True)
    with serve(make_app(lm)) as client:
        with client.stream("POST", "/generate", json={"prompt": "hi", "max_tokens": 3}) as response:
            events = iter_sse(response)
            lm.release(1)  # from the test's thread, while the server's loop runs the stream
            assert next(events).json() == {"piece": "Hello"}
            # The second piece has not been produced: the server is streaming, not buffering.
            assert lm.calls[0].pieces == ["Hello"]
            assert client.get("/status").json() == {"active": 1}
            lm.release()
            rest = [event.data for event in events]
        assert rest == ['{"piece": ","}', '{"piece": " world"}', "[DONE]"]
        assert lm.wait_closed() and lm.calls[0].finish_reason == "length"
        assert client.get("/status").json() == {"active": 0}
    assert lm.calls[0].prompt == "hi" and not lm.stalled


def test_wait_closed_times_out_while_a_stream_is_open():
    lm = ScriptedLM(SCRIPT)
    assert not lm.wait_closed(0.0)  # nothing was ever started
    stream = lm.stream("p")
    assert not lm.wait_closed(0.01)
    threading.Timer(0.02, lambda: asyncio.run(stream.aclose())).start()
    assert lm.wait_closed(2.0)
