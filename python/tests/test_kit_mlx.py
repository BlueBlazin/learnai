"""The MLX kit: MLXTinyLM, EvalSpy, TraceCounter, mlx_memory; and CountingFile."""

import textwrap

import numpy as np
import pytest

from whetstone.kit import CountingFile, TraceCounter

mx = pytest.importorskip("mlx.core")

from whetstone.kit import EvalSpy, MLXTinyLM, mlx_memory

SOLUTION = textwrap.dedent(
    """
    import mlx.core as mx
    import numpy as np
    from mlx.core import eval as force


    def one_eval_per_token(lm, prompt, n):
        logits, cache = lm(mx.array(prompt))
        token = mx.argmax(logits[-1])
        out = []
        for _ in range(n):
            mx.eval(token, cache)
            out.append(token)
            logits, cache = lm(token[None], cache)
            token = mx.argmax(logits[-1])
        return out


    def reads_everywhere(x):
        a = x.sum().item()
        b = x.tolist()
        c = np.array(x)
        d = float(x[0])
        print(x[0])
        read = x[1].item
        e = read()
        return a, b, c, d, e


    def aliased_and_async(x):
        force(x)
        mx.async_eval(x * 2)
        return x


    def streaming(lm, prompt, n):
        logits, cache = lm(mx.array(prompt))
        token = mx.argmax(logits[-1])
        mx.async_eval(token, cache)
        for _ in range(n):
            logits, cache = lm(token[None], cache)
            following = mx.argmax(logits[-1])
            mx.async_eval(following, cache)
            yield token.item()
            token = following


    def compiled_step(lm):
        def step(token, state):
            logits, state = lm.step(token, state)
            return mx.argmax(logits, axis=-1).astype(mx.int32), state

        return mx.compile(step)
    """
)


@pytest.fixture
def solution(tmp_path):
    path = tmp_path / "mlx.py"
    path.write_text(SOLUTION)
    namespace: dict = {}
    exec(compile(SOLUTION, str(path), "exec"), namespace)  # noqa: S102
    namespace["__file__"] = str(path)
    return namespace


# -- MLXTinyLM ------------------------------------------------------------------------------------


def test_model_is_seeded_lazy_and_a_module():
    from mlx import nn

    lm = MLXTinyLM(seed=3, n_kv_heads=2)
    assert isinstance(lm, nn.Module)
    logits, cache = lm(mx.array([1, 2, 3]))
    assert logits.shape == (3, 64) and logits.dtype == mx.float32
    assert len(cache) == 2 and cache[0][0].shape == (1, 2, 3, 8)
    again, _ = MLXTinyLM(seed=3, n_kv_heads=2)(mx.array([1, 2, 3]))
    other, _ = MLXTinyLM(seed=4, n_kv_heads=2)(mx.array([1, 2, 3]))
    assert mx.array_equal(logits, again) and not mx.array_equal(logits, other)
    assert 0.5 < float(logits.std()) < 6.0
    names = [name for name, _ in nn.utils.tree_flatten(lm.parameters())]
    assert "embed.weight" in names and "layers.1.q_proj.weight" in names
    assert lm.counters.snapshot() == {"forward_calls": 1, "tokens_processed": 3}
    with pytest.raises(ValueError, match="n_heads a multiple of n_kv_heads"):
        MLXTinyLM(n_heads=4, n_kv_heads=3)


@pytest.mark.parametrize("options", [{}, {"n_kv_heads": 1}, {"rope": False}, {"n_layers": 3}])
def test_cache_batch_and_fixed_shape_step_agree_with_one_full_call(options):
    lm = MLXTinyLM(seed=1, **options)
    tokens = mx.array([5, 9, 2, 40, 7, 7, 13])
    full, _ = lm(tokens)
    head, cache = lm(tokens[:3])
    tail, cache = lm(tokens[3:], cache)
    assert float(mx.abs(mx.concatenate([head, tail]) - full).max()) < 1e-5
    assert cache[0][0].shape[2] == 7
    batch, _ = lm(mx.stack([tokens, tokens[::-1]]))
    assert batch.shape == (2, 7, 64) and float(mx.abs(batch[0] - full).max()) < 1e-5
    state = lm.init_state(max_len=16)
    rows = []
    for token in tokens.tolist():
        logits, state = lm.step(mx.array([token]), state)
        rows.append(logits[0])
    assert float(mx.abs(mx.stack(rows) - full).max()) < 1e-5
    assert int(state["length"]) == 7 and state["keys"][0].shape == (1, lm.n_kv_heads, 16, 8)


def test_mask_and_positions_make_left_padded_rows_match_single_sequences():
    lm = MLXTinyLM(seed=2)
    short, long = [3, 4], [9, 8, 7, 6]
    alone, _ = lm(mx.array(short))
    padded = mx.array([[0, 0, *short], long])
    # Row 0 has two pads on the left: they are masked out and positions start after them.
    allowed = np.tril(np.ones((4, 4), dtype=bool))[None].repeat(2, axis=0)
    allowed[0, :, :2] = False
    allowed[0, 0, 0] = allowed[0, 1, 1] = True  # a pad row must see something
    mask = mx.array(np.where(allowed, 0.0, -np.inf).astype(np.float32))[:, None]
    positions = mx.array([[0, 0, 0, 1], [0, 1, 2, 3]])
    batch, _ = lm(padded, mask=mask, positions=positions)
    assert float(mx.abs(batch[0, 2:] - alone).max()) < 1e-5
    unmasked, _ = lm(padded)
    assert float(mx.abs(unmasked[0, 2:] - alone).max()) > 1e-3


def test_half_precision_model_stays_half_precision():
    lm = MLXTinyLM(seed=1, dtype=mx.float16)
    logits, cache = lm(mx.array([1, 2, 3]))
    assert logits.dtype == mx.float16 and cache[0][0].dtype == mx.float16
    logits, state = lm.step(mx.array([1]), lm.init_state(8))
    assert logits.dtype == mx.float16 and state["keys"][0].dtype == mx.float16


def test_compiled_step_is_traced_once_and_equals_the_plain_loop(solution):
    lm = MLXTinyLM(seed=1)
    plain_state, token, want = lm.init_state(64), mx.array([3], dtype=mx.int32), []
    for _ in range(40):
        logits, plain_state = lm.step(token, plain_state)
        token = mx.argmax(logits, axis=-1).astype(mx.int32)
        want.append(int(token[0]))
    lm.counters.reset()
    step = solution["compiled_step"](lm)
    state, token, got = lm.init_state(64), mx.array([3], dtype=mx.int32), []
    for _ in range(40):
        token, state = step(token, state)
        mx.eval(token, state)
        got.append(int(token[0]))
    assert got == want
    # The Python body ran once: forward_calls counts traces under mx.compile.
    assert lm.counters.forward_calls == 1


# -- TraceCounter -----------------------------------------------------------------------------------


def test_trace_counter_counts_runs_of_the_python_body():
    tc = TraceCounter()
    fast = mx.compile(lambda x: tc.mark(x) * 2)
    for value in range(10):
        assert float(fast(mx.array(float(value)))) == 2.0 * value
    assert tc.count == 1
    fast(mx.zeros((3,)))  # a new shape is a new trace
    assert tc.count == 2
    plain = tc.wrap(lambda x: x + 1)
    assert float(plain(mx.array(1.0))) == 2.0 and tc.count == 3
    tc.reset()
    assert tc.count == 0


# -- EvalSpy ------------------------------------------------------------------------------------------


def test_spy_counts_explicit_evaluations_made_by_the_solution(solution):
    lm = MLXTinyLM(seed=1)
    with EvalSpy(file=solution["__file__"]) as spy:
        tokens = solution["one_eval_per_token"](lm, [1, 2, 3], 5)
        mx.eval(tokens)  # the test's own evaluation is not the solution's
        int(tokens[0])
    assert (spy.evals, spy.async_evals, spy.implicit) == (5, 0, 0)
    assert {event.what for event in spy.events} == {"mx.eval"}


def test_spy_sees_implicit_reads_with_their_lines(solution, capsys):
    x = mx.arange(4) * 1.5
    with EvalSpy(file=solution["__file__"]) as spy:
        solution["reads_everywhere"](x)
    capsys.readouterr()
    assert spy.evals == 0 and spy.implicit == 6
    assert [event.what for event in spy.events] == [
        "item", "tolist", "np.array", "float", "print", "item",
    ]  # fmt: skip
    first = SOLUTION.splitlines().index("    a = x.sum().item()") + 1
    assert [event.line for event in spy.events] == [first, first + 1, first + 2, first + 3,
                                                    first + 4, first + 6]  # fmt: skip
    assert spy.describe().splitlines()[0] == f"line {first}: read (item)"


def test_spy_sees_aliases_and_the_order_of_async_evaluation(solution):
    with EvalSpy(file=solution["__file__"]) as spy:
        solution["aliased_and_async"](mx.ones((2,)))
    assert spy.kinds() == ["eval", "async_eval"]

    lm = MLXTinyLM(seed=1)
    with EvalSpy(file=solution["__file__"]) as spy:
        tokens = list(solution["streaming"](lm, [1, 2, 3], 4))
    assert len(tokens) == 4 and spy.evals == 0
    # Step t + 1 is scheduled before token t is read, every time.
    assert spy.kinds() == ["async_eval"] + ["async_eval", "read"] * 4


def test_spies_can_follow_one_another_and_need_a_file():
    with EvalSpy(file=__file__) as first:
        mx.eval(mx.ones(2))
    with EvalSpy(file=__file__) as second:
        mx.eval(mx.ones(2))
        mx.eval(mx.ones(2))
    assert (first.evals, second.evals) == (1, 2)
    with pytest.raises(ValueError, match="needs file="):
        EvalSpy()


# -- mlx_memory -----------------------------------------------------------------------------------------


def test_memory_measures_what_the_block_allocates_beyond_its_inputs():
    x = mx.random.normal((1024, 1024))  # 4 MiB, allocated before the block
    mx.eval(x)
    with mlx_memory() as mem:
        y = x @ x
        z = y + 1
        mx.eval(y, z)
    assert 7.9 <= mem.peak_mib <= 9.0 and 7.9 <= mem.active_mib <= 9.0
    assert mem.baseline_bytes >= x.nbytes
    del y, z

    with mlx_memory() as lazy:
        x @ x  # never evaluated: nothing is allocated
    assert lazy.peak_bytes == 0

    with mlx_memory() as freed:
        total = (x @ x).sum()
        mx.eval(total)
    # The 4 MiB product was needed and is gone again: it shows in the peak only.
    assert freed.peak_mib >= 3.9 and freed.active_mib < 1.0
    assert "peak=" in repr(freed)


def test_memory_tells_float16_from_float32():
    x16 = mx.random.normal((1024, 1024)).astype(mx.float16)
    mx.eval(x16)
    with mlx_memory() as half:
        out = x16 * 2.0  # a Python scalar does not promote
        mx.eval(out)
    with mlx_memory() as single:
        promoted = x16 * mx.array(2.0)  # a float32 array does
        mx.eval(promoted)
    assert promoted.dtype == mx.float32 and out.dtype == mx.float16
    assert half.peak_mib < 3.0 < single.peak_mib


# -- CountingFile ------------------------------------------------------------------------------------------


def test_counting_file_charges_for_the_ranges_asked_for(tmp_path):
    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 400  # 102400 bytes: seven 16 KiB pages
    path.write_bytes(payload)
    with CountingFile(path) as f:
        assert len(f) == len(payload)
        head = f[:8]
        assert isinstance(head, memoryview) and bytes(head) == payload[:8]
        values = np.frombuffer(f[40000:40016], dtype=np.uint8)  # zero-copy
        assert values.tolist() == list(payload[40000:40016])
        assert f[3] == 3 and f[-1] == 255
        assert f.counters.snapshot() == {"bytes_read": 8 + 16 + 1 + 1, "reads": 4}
        assert f.pages_touched == 3  # pages 0, 2, and 6
        f.seek(100)
        assert f.read(10) == payload[100:110] and f.tell() == 110
        buffer = bytearray(4)
        assert f.readinto(buffer) == 4 and bytes(buffer) == payload[110:114]
        f.seek(-2, 2)
        assert f.read() == payload[-2:] and f.read(5) == b""
        assert f.counters.bytes_read == 26 + 10 + 4 + 2
        with pytest.raises(ValueError, match="contiguous"):
            f[::2]
    whole = CountingFile(path)
    whole.read()
    assert whole.counters.bytes_read == len(payload) and whole.pages_touched == 7
    assert CountingFile(path, page_size=4096).page_size == 4096


# -- the speed judge on compiled MLX functions ------------------------------------------------------------


def test_compare_speed_times_a_compiled_step_after_warmup(solution):
    """The example of the authoring guide (section 5.5): build outside the timer, let
    warmup absorb the trace, force what the callable returns."""
    from conftest import fixture_problem

    from whetstone.judge.context import make_context
    from whetstone.judge.registry import Case
    from whetstone.problem import load_problem

    case = Case(id="unit", name="unit", public=True, fn=lambda ctx: None)
    ctx = make_context(
        load_problem(fixture_problem("unit-normalize")), solution=None, framework="mlx", case=case
    )
    models = []

    def loop(make_step):
        lm = MLXTinyLM(seed=5)
        models.append(lm)
        step = make_step(lm)

        def run():
            token, state = mx.array([3], dtype=mx.int32), lm.init_state(max_len=64)
            for _ in range(32):
                token, state = step(token, state)
            return token, state

        return run

    with ctx.active():
        ratio = ctx.compare_speed(
            loop(solution["compiled_step"]),
            loop(solution["compiled_step"]),
            max_ratio=10.0,
            label="32 compiled steps",
            tokens=32,
            rounds=3,
        )
    assert 0.1 < ratio < 10.0
    metrics = {m["name"]: m for m in ctx.metrics}
    assert metrics["slowdown"]["limit"] == 10 and metrics["tokens_per_s"]["value"] > 0
    # However many times the loop ran, each compiled step was traced exactly once.
    assert [lm.counters.forward_calls for lm in models] == [1, 1]
