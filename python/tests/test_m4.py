"""Milestone 4: fixes reported by milestone 3 authors, CUDA detection, `learnai doctor`,
the JAX compilation cache, and the small kit helpers."""

import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from conftest import FIXTURES, run_harness

from learnai import cli, hardware, runner
from learnai import validate as gate
from learnai.judge import frameworks
from learnai.kit import RealConfig, logit_kl
from learnai.problem import load_problem
from learnai.tracebacks import exception_line, hidden_error_message

M3 = FIXTURES / "m3"
CUDA_SCALE = M3 / "problems" / "gpu" / "cuda-scale"
MLX_SCALE = M3 / "problems" / "gpu" / "mlx-scale"


# -- one-line exception messages ----------------------------------------------------------


class Custom(Exception):
    pass


@pytest.mark.parametrize(
    ("exc", "line"),
    [
        (ValueError("one line"), "ValueError: one line"),
        (RuntimeError("shapes differ\n  lhs: (3, 4)\n  rhs: (5,)"), "RuntimeError: shapes differ"),
        (RuntimeError("\n\n   spaced   out \nmore"), "RuntimeError: spaced out"),
        (RuntimeError("Error(s) in loading:\n\tMissing key x\nhint"),
         "RuntimeError: Error(s) in loading: Missing key x"),
        (RuntimeError("-----\n=====\nthe real message"), "RuntimeError: the real message"),
        (KeyError("k"), "KeyError: 'k'"),
        (ValueError(""), "ValueError"),
        (AssertionError(), "AssertionError"),
        (SystemExit(3), "SystemExit: 3"),
        (Custom("mine\nand more"), "test_m4.Custom: mine"),
    ],
)  # fmt: skip
def test_exception_line_is_the_type_and_the_first_meaningful_line(exc, line):
    got = exception_line(exc)
    assert got.endswith(line.removeprefix("test_m4.")) and "\n" not in got
    assert got.split(":")[0].split(".")[-1] == type(exc).__name__


def test_exception_line_keeps_the_message_of_a_syntax_error():
    with pytest.raises(SyntaxError) as failure:
        compile("def f(:\n    pass\n", "solution.py", "exec")
    assert exception_line(failure.value) == "SyntaxError: invalid syntax"


def test_exception_line_survives_a_broken_str_and_notes():
    class Broken(Exception):
        def __str__(self):
            raise RuntimeError("no")

    assert exception_line(Broken()).endswith("Broken")
    noted = ValueError("first")
    noted.add_note("a note that is not the message")
    assert exception_line(noted) == "ValueError: first"


def test_a_multi_line_exception_gives_an_informative_one_line_case_message(tmp_path):
    # Through the harness: a hidden case keeps one line (type, first line, location).
    problem = tmp_path / "problems" / "basics" / "sum-list"
    shutil.copytree(FIXTURES / "catalog" / "problems" / "basics" / "sum-list", problem)
    shutil.copy(FIXTURES / "catalog" / "problems" / "basics" / "track.toml", problem.parent)
    solution = tmp_path / "solution.py"
    solution.write_text(
        "def total(values):\n"
        "    raise RuntimeError('size mismatch for layer 3\\n  expected: (4, 8)\\n  got: (4, 9)')\n"
    )
    run = run_harness(problem, solution)
    messages = {e["id"]: e["message"] for e in run.of("case_end")}
    public = next(e for e in run.start["cases"] if e["public"])["id"]
    hidden = next(e for e in run.start["cases"] if not e["public"])["id"]
    assert messages[public] == "RuntimeError: size mismatch for layer 3"
    assert messages[hidden] == (
        "RuntimeError: size mismatch for layer 3 (solution.py line 2, in total)"
    )
    assert "expected: (4, 8)" in run.case(public)["detail"]  # the full text is in the detail
    assert run.case(hidden)["detail"] is None


def test_hidden_error_message_is_cut_and_located(tmp_path):
    source = tmp_path / "solution.py"
    source.write_text("def f():\n    raise ValueError('x' * 500 + '\\nsecond line')\n")
    spec = importlib.util.spec_from_file_location("hidden_error_solution", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError) as failure:
        module.f()
    message = hidden_error_message(failure.value, str(source))
    assert message.endswith("... (solution.py line 2, in f)") and len(message) < 240
    assert "second line" not in message


# -- validate on a machine without the hardware --------------------------------------------


def test_validate_does_not_claim_skips_when_every_case_ran(monkeypatch, capsys):
    """An mlx-gpu problem none of whose cases is marked: with the hardware pretended
    away every case still runs. The warning must not speak of skipped cases."""
    pytest.importorskip("mlx.core")
    root = M3
    monkeypatch.setenv("LEARNAI_ROOT", str(root))
    monkeypatch.setenv("LEARNAI_HARDWARE", "cpu")
    problem = load_problem(MLX_SCALE)
    command = runner.harness_command(
        MLX_SCALE, problem.reference_path("mlx"), "mlx", "submit", reveal=True
    )
    result = runner.run(command)
    marked = [c for c in result.start["cases"] if c.get("requires")]
    (report,) = gate.validate(root, [MLX_SCALE])
    assert report.errors == [] and report.status == "ok"
    text = " ".join(report.warnings)
    assert "0 were NOT validated" not in text and "cases were skipped" not in text
    if marked:
        assert f"{len(marked)} {'was' if len(marked) == 1 else 'were'} NOT validated" in text
        assert "GPU cases not validated:" in text
    else:
        assert "ran and passed, but not on the hardware the problem requires" in text


def _judge(job_kind, end, cases, problem):
    report = gate.Report(problem.id, problem.path)
    job = gate._Job(job_kind, "torch", Path("x.py"))
    start = {"cases": [{"id": c["id"], "public": i == 0} for i, c in enumerate(cases)]}
    result = runner.RunResult(start=start, cases=cases, end=end)
    gate._judge(report, problem, job, result)
    return report


def test_validate_wording_for_runs_that_lack_only_the_required_hardware():
    problem = load_problem(CUDA_SCALE)
    passed = [{"id": f"c{i}", "status": "pass", "duration_ms": 1.0} for i in range(3)]
    end = {"status": "hardware_missing", "passed": 3, "skipped": 0, "duration_ms": 5.0,
           "missing_hardware": "cuda"}  # fmt: skip

    report = _judge("reference", end, passed, problem)
    assert report.errors == [] and report.ran
    assert report.warnings == [
        (
            "reference (torch): all 3 cases ran and passed, but not on the hardware the problem "
            "requires (an NVIDIA GPU (CUDA)), which this machine does not have"
        )
    ]
    # A mutant or a starter that passes every case is caught: nothing was skipped.
    assert _judge("mutant", end, passed, problem).errors == [
        "mutant x: passed every case; no test catches this bug. Add a case that does."
    ]
    assert _judge("starter", end, passed, problem).errors == [
        "starter (torch): passes every case; the tests must reject it"
    ]

    skipped = {
        "id": "gpu_speed",
        "status": "skipped",
        "duration_ms": 0,
        "message": "skipped: needs an NVIDIA GPU (CUDA), which this machine does not have",
    }
    end = {**end, "passed": 2, "skipped": 1}
    report = _judge("reference", end, [*passed[:2], skipped], problem)
    assert report.warnings == [
        (
            "reference (torch): only 2 of 3 cases could run on this machine (all passed); 1 was NOT "
            "validated: needs an NVIDIA GPU (CUDA), which this machine does not have "
            "(GPU cases not validated: gpu_speed)"
        )
    ]
    assert (
        _judge("mutant", end, [*passed[:2], skipped], problem)
        .warnings[0]
        .startswith(
            "mutant x: NOT validated: no case that can run on this machine catches it, and 1"
        )
    )


def test_validate_summary_counts_problems_with_unvalidated_cases(monkeypatch, capsys):
    monkeypatch.setenv("LEARNAI_ROOT", str(M3))
    monkeypatch.setenv("LEARNAI_HARDWARE", "cpu")
    assert cli.main(["validate", "cuda-scale"]) == 0
    out = capsys.readouterr().out
    assert "(GPU cases not validated: on_cuda)" in out
    assert out.strip().splitlines()[-1] == (
        "1 of 1 problems passed validation; 1 passed with cases that could NOT be validated "
        "on this machine (see the warnings above)"
    )


# -- RealConfig ---------------------------------------------------------------------------

GRANITE_RAW = {
    "model_type": "granitemoehybrid", "num_hidden_layers": 2, "hidden_size": 64,
    "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 96,
    "vocab_size": 128, "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
    "rope_theta": 10000000, "attention_multiplier": 0.0625, "embedding_multiplier": 12,
    "residual_multiplier": 0.263, "logits_scaling": 4, "tie_word_embeddings": True,
    "num_local_experts": 0, "layer_types": ["attention", "attention"],
    "torch_dtype": "bfloat16", "transformers_version": "4.56.0",
    "bos_token_id": 100, "eos_token_id": 100, "pad_token_id": 101,
}  # fmt: skip


def test_realconfig_accepts_a_config_saved_again_by_transformers_5():
    original = RealConfig.from_json(GRANITE_RAW)
    # What `transformers` 5 writes: other layer names, rope_parameters, dtype.
    saved = {k: v for k, v in GRANITE_RAW.items() if k not in ("rope_theta", "torch_dtype")}
    saved |= {
        "layer_types": ["full_attention", "full_attention"],
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "dtype": "bfloat16",
        "transformers_version": "5.23.0",
    }
    assert RealConfig.from_json(saved) == original
    assert RealConfig.from_json({**saved, "layer_types": "full_attention"}) == original
    assert RealConfig.from_json({**GRANITE_RAW, "layer_types": "attention"}) == original
    for other in (["full_attention", "mamba"], "mamba"):
        with pytest.raises(ValueError, match="non-attention layers"):
            RealConfig.from_json({**saved, "layer_types": other})
    with pytest.raises(ValueError, match="experts"):
        RealConfig.from_json({**saved, "num_local_experts": 4})


def test_realconfig_round_trips_through_transformers_5(tmp_path):
    transformers = pytest.importorskip("transformers")
    (tmp_path / "config.json").write_text(json.dumps(GRANITE_RAW))
    try:
        config = transformers.AutoConfig.from_pretrained(tmp_path)
    except Exception as exc:  # noqa: BLE001 - an older transformers without this model type
        pytest.skip(f"transformers cannot load the config: {exc}")
    saved = json.loads(config.to_json_string(use_diff=False))
    assert RealConfig.from_json(saved) == RealConfig.from_json(GRANITE_RAW)


# -- CUDA detection (mocked: this machine has no NVIDIA GPU) -------------------------------


@pytest.fixture
def fresh_detection(monkeypatch):
    monkeypatch.delenv("LEARNAI_HARDWARE", raising=False)
    hardware._detect.cache_clear()
    yield
    hardware._detect.cache_clear()


def fake_cuda(monkeypatch, names=("NVIDIA RTX 4090",), memory=24 * 2**30):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(names))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: names[index or 0])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index=None: types.SimpleNamespace(total_memory=memory),
    )
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising=False)


def test_cuda_is_detected_through_torch(monkeypatch, fresh_detection):
    assert hardware.available("cuda") is False
    assert hardware.cuda_devices() == []
    hardware._detect.cache_clear()
    fake_cuda(monkeypatch, names=("NVIDIA RTX 4090", "NVIDIA T4"))
    assert hardware.available("cuda") is True
    assert hardware.cuda_devices() == [
        {"index": 0, "name": "NVIDIA RTX 4090", "memory_bytes": 24 * 2**30},
        {"index": 1, "name": "NVIDIA T4", "memory_bytes": 24 * 2**30},
    ]
    report = hardware.report()["cuda"]
    assert report["present"] is True and report["name"] == "NVIDIA RTX 4090"
    assert report["memory_bytes"] == 24 * 2**30 and report["torch_cuda_version"] == "13.0"
    assert len(report["devices"]) == 2


def test_cuda_detection_survives_a_broken_driver_or_missing_torch(monkeypatch, fresh_detection):
    import torch

    def broken():
        raise RuntimeError("CUDA driver initialization failed")

    monkeypatch.setattr(torch.cuda, "is_available", broken)
    assert hardware.available("cuda") is False and hardware.cuda_devices() == []
    hardware._detect.cache_clear()

    # A device that cannot be described is still a device.
    fake_cuda(monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: broken())
    assert hardware.cuda_devices() == [{"index": 0, "name": None, "memory_bytes": None}]
    hardware._detect.cache_clear()

    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` raises ImportError
    assert hardware.available("cuda") is False
    assert hardware.report()["cuda"] == {
        "present": False, "name": None, "memory_bytes": None, "devices": [],
        "torch_cuda_version": None,
    }  # fmt: skip


def test_the_override_wins_over_detection_but_not_in_the_report(monkeypatch, fresh_detection):
    monkeypatch.setenv("LEARNAI_HARDWARE", "cuda")
    assert hardware.available("cuda") and not hardware.available("mlx-gpu")
    assert hardware.report()["cuda"]["present"] is False  # what the machine really has


def test_cuda_problem_ends_hardware_missing_when_every_cpu_case_passes():
    env = {"LEARNAI_HARDWARE": "cpu"}
    run = run_harness(CUDA_SCALE, CUDA_SCALE / "reference" / "torch.py", "torch", env=env)
    assert run.returncode == 0
    assert sorted(run.statuses.values()) == ["pass", "pass", "skipped"]
    assert run.end["status"] == "hardware_missing" and run.end["missing_hardware"] == "cuda"
    assert (run.end["passed"], run.end["total"], run.end["skipped"]) == (2, 3, 1)
    # The public cases alone: nothing is skipped, and it is still not `accepted`.
    run = run_harness(CUDA_SCALE, CUDA_SCALE / "reference" / "torch.py", "torch", "run", env=env)
    assert run.end["status"] == "hardware_missing" and run.end["skipped"] == 0
    # A wrong solution fails on the CPU cases: the failure decides, not the skip.
    mutant = CUDA_SCALE / "mutants" / "ignores_factor.py"
    run = run_harness(CUDA_SCALE, mutant, "torch", env=env)
    assert run.end["status"] == "wrong_answer"


CUDA_KIT_TESTS = """
import numpy as np
import torch

from learnai.judge import case, check
from learnai.kit import DeviceSpy, FakeDevice, SyncSpy
from learnai.kit import tritonshim as tl


@case("no host synchronization, no tensor without a device", public=True)
def quiet(ctx):
    x = torch.tensor([1.0, -2.0, 0.5])
    with SyncSpy() as spy, DeviceSpy() as devices:  # both find the solution file themselves
        out = ctx.solution.scale(x, 3.0, torch.device("cpu"))
    check.that(spy.syncs == 0, f"scale: {spy.syncs} host synchronizations", spy.describe())
    check.that(not devices.implicit, "scale: a tensor was created without device=",
               devices.describe())
    check.allclose(out.numpy(), np.array([3.0, -6.0, 1.5], dtype=np.float32), label="scale")


@case("the kit device and the Triton shim are usable from a case")
def kit(ctx):
    dev = FakeDevice()
    dev.kernel(0.5)()
    dev.sync()
    check.allclose(dev.clock(), 0.5, rtol=1e-9, label="virtual time")

    @tl.jit
    def double(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n, other=0.0) * 2, mask=offs < n)

    x, out = torch.arange(5, dtype=torch.float32), torch.zeros(5)
    tl.launch(double, (2,), x, out, 5, BLOCK=4)
    check.equal(out.tolist(), [0.0, 2.0, 4.0, 6.0, 8.0], label="doubled")


@case("the result is on the CUDA device", requires="cuda")
def on_cuda(ctx):
    out = ctx.solution.scale(torch.ones(3), 2.0, torch.device("cuda"))
    check.that(out.device.type == "cuda", f"scale: the result is on {out.device.type}")
"""


def test_the_cuda_kit_works_inside_the_harness(tmp_path):
    """A CUDA-track problem end to end on this machine: the spies find the solution
    file on their own, a mutant is caught on the CPU, the GPU case is skipped."""
    problem = tmp_path / "problems" / "gpu" / "cuda-scale"
    shutil.copytree(CUDA_SCALE, problem)
    shutil.copy(CUDA_SCALE.parent / "track.toml", problem.parent)
    (problem / "tests.py").write_text(CUDA_KIT_TESTS)
    env = {"LEARNAI_HARDWARE": "cpu"}
    run = run_harness(problem, problem / "reference" / "torch.py", "torch", env=env)
    assert run.end["status"] == "hardware_missing", run.events
    assert (run.end["passed"], run.end["skipped"]) == (2, 1)

    noisy = tmp_path / "noisy.py"
    noisy.write_text(
        "import torch\n\n\n"
        "def scale(x, factor, device):\n"
        "    if x.sum() > 100:\n"
        "        return x\n"
        "    ramp = torch.arange(x.shape[0])\n"
        "    return (x * factor + 0 * ramp).to(device)\n"
    )
    run = run_harness(problem, noisy, "torch", env=env)
    assert run.end["status"] == "wrong_answer"
    failed = run.case("quiet")
    assert failed["message"] == "scale: 1 host synchronizations"
    assert "line 5: read (bool)" in failed["detail"]


# -- learnai doctor ----------------------------------------------------------------------------


def test_doctor_json_has_the_documented_schema(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("LEARNAI_ROOT", str(tmp_path))
    monkeypatch.delenv("LEARNAI_MODELS_DIR", raising=False)
    monkeypatch.delenv("LEARNAI_HARDWARE", raising=False)
    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert list(report) == [
        "schema", "learnai", "python", "executable", "platform", "frameworks", "packages",
        "hardware", "hardware_override", "root", "models_dir",
    ]  # fmt: skip
    assert report["schema"] == 1
    assert report["python"].startswith("3.") and report["executable"] == sys.executable
    assert set(report["platform"]) == {"system", "machine", "release"}
    assert set(report["frameworks"]) == {"torch", "jax", "mlx"}
    assert report["frameworks"]["torch"]
    assert {"numpy", "triton", "transformers", "mlx-lm"} <= set(report["packages"])
    assert list(report["hardware"]) == ["cpu", "mlx-gpu", "cuda"]
    assert report["hardware"]["cpu"]["present"] is True
    assert set(report["hardware"]["cuda"]) == {
        "present", "name", "memory_bytes", "devices", "torch_cuda_version",
    }  # fmt: skip
    assert set(report["hardware"]["mlx-gpu"]) == {"present", "name", "memory_bytes"}
    assert report["hardware_override"] is None
    assert report["root"] == str(tmp_path.resolve())
    assert report["models_dir"] == str(tmp_path.resolve() / ".learnai" / "models")

    monkeypatch.setenv("LEARNAI_HARDWARE", "cuda")
    monkeypatch.setenv("LEARNAI_MODELS_DIR", str(tmp_path / "elsewhere"))
    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["hardware_override"] == ["cpu", "cuda"]
    assert report["models_dir"] == str((tmp_path / "elsewhere").resolve())
    assert cli.main(["doctor"]) == 0
    assert "LEARNAI_HARDWARE pretends: cpu, cuda" in capsys.readouterr().out


def test_doctor_reports_cuda_and_never_fails_on_missing_packages(capsys, monkeypatch, tmp_path):
    from importlib import metadata

    monkeypatch.setenv("LEARNAI_ROOT", str(tmp_path))
    monkeypatch.delenv("LEARNAI_HARDWARE", raising=False)
    hardware._detect.cache_clear()
    fake_cuda(monkeypatch)

    def nothing_installed(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", nothing_installed)
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    try:
        assert cli.main(["doctor", "--json"]) == 0
    finally:
        hardware._detect.cache_clear()
    report = json.loads(capsys.readouterr().out)
    assert set(report["frameworks"].values()) == {None}
    assert set(report["packages"].values()) == {None}
    assert report["hardware"]["mlx-gpu"] == {"present": False, "name": None, "memory_bytes": None}
    cuda = report["hardware"]["cuda"]
    assert (cuda["present"], cuda["name"], cuda["memory_bytes"]) == (
        True, "NVIDIA RTX 4090", 24 * 2**30,
    )  # fmt: skip


def test_doctor_imports_no_framework_it_does_not_need():
    # A fresh interpreter: `learnai doctor --json` must not import jax, transformers, or the kit.
    code = (
        "import sys, json, io, contextlib\n"
        "from learnai import cli\n"
        "out = io.StringIO()\n"
        "with contextlib.redirect_stdout(out):\n"
        "    assert cli.main(['doctor', '--json']) == 0\n"
        "json.loads(out.getvalue())\n"
        "heavy = [m for m in ('jax', 'transformers', 'learnai.kit', 'learnai.judge') "
        "if m in sys.modules]\n"
        "print(heavy)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code], capture_output=True, text=True, check=False, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


# -- the JAX compilation cache ------------------------------------------------------------


def test_jax_cache_is_configured_tidied_and_can_be_switched_off(tmp_path, monkeypatch):
    jax = pytest.importorskip("jax")
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    directory = tmp_path / "cache" / "jax"
    directory.mkdir(parents=True)
    (directory / "torn-cache").write_bytes(b"")  # what a killed worker leaves behind
    (directory / "good-cache").write_bytes(b"x" * 10)
    before = {
        name: getattr(jax.config, name)
        for name in ("jax_compilation_cache_dir", "jax_persistent_cache_min_compile_time_secs",
                     "jax_persistent_cache_min_entry_size_bytes")
    }  # fmt: skip
    try:
        monkeypatch.setenv("LEARNAI_JAX_CACHE", "off")
        assert frameworks.use_jax_compilation_cache(directory) is False
        monkeypatch.delenv("LEARNAI_JAX_CACHE")
        assert frameworks.use_jax_compilation_cache(directory) is True
        assert jax.config.jax_compilation_cache_dir == str(directory)
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
        assert sorted(p.name for p in directory.iterdir()) == ["good-cache"]

        other = tmp_path / "elsewhere"
        monkeypatch.setenv("LEARNAI_JAX_CACHE", str(other))
        assert frameworks.use_jax_compilation_cache(directory) is True
        assert jax.config.jax_compilation_cache_dir == str(other) and other.is_dir()

        # A cache the user configured is theirs.
        monkeypatch.delenv("LEARNAI_JAX_CACHE")
        monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "mine"))
        assert frameworks.use_jax_compilation_cache(directory) is False

        # Beyond the bound the cache is emptied; a failure to set up is not an error.
        monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR")
        monkeypatch.setattr(frameworks, "JAX_CACHE_MAX_BYTES", 5)
        assert frameworks.use_jax_compilation_cache(directory) is True
        assert list(directory.iterdir()) == []
        blocked = tmp_path / "file"
        blocked.write_text("not a directory")
        assert frameworks.use_jax_compilation_cache(blocked / "jax") is False
    finally:
        for name, value in before.items():
            jax.config.update(name, value)


def test_a_jax_harness_run_fills_the_cache_and_the_next_run_uses_it(tmp_path):
    pytest.importorskip("jax")
    problem = FIXTURES / "catalog" / "problems" / "judges" / "unit-normalize"
    cache = tmp_path / "jax-cache"
    env = {"LEARNAI_JAX_CACHE": str(cache)}
    first = run_harness(problem, problem / "reference" / "jax.py", "jax", env=env)
    assert first.end["status"] == "accepted"
    entries = sorted(p.name for p in cache.iterdir())
    assert entries, "the first run wrote no compiled kernels"
    second = run_harness(problem, problem / "reference" / "jax.py", "jax", env=env)
    assert second.end["status"] == "accepted"
    assert sorted(p.name for p in cache.iterdir()) == entries  # nothing new to compile
    assert first.statuses == second.statuses
    disabled = run_harness(
        problem, problem / "reference" / "jax.py", "jax", env={"LEARNAI_JAX_CACHE": "0"}
    )
    assert disabled.end["status"] == "accepted"


# -- small kit helpers --------------------------------------------------------------------


def test_logit_kl_is_the_kl_divergence_of_the_two_softmaxes():
    rng = np.random.default_rng(0)
    p, q = rng.standard_normal((5, 16)) * 4, rng.standard_normal((5, 16)) * 4
    kl = logit_kl(p, q)
    P = np.exp(p) / np.exp(p).sum(-1, keepdims=True)
    Q = np.exp(q) / np.exp(q).sum(-1, keepdims=True)
    np.testing.assert_allclose(kl, (P * np.log(P / Q)).sum(-1), rtol=1e-10)
    assert kl.shape == (5,) and (kl > 0).all()
    np.testing.assert_allclose(logit_kl(p, p), 0.0, atol=1e-12)
    np.testing.assert_allclose(logit_kl(p + 7.0, q - 3.0), kl, rtol=1e-10)  # shift invariant
    # Float32 logits of large magnitude neither overflow nor lose the answer.
    big = np.array([[1000.0, 0.0, -1000.0]], dtype=np.float32)
    assert np.isfinite(logit_kl(big, big[:, ::-1])).all()
    assert logit_kl(p.T, q.T, axis=0).shape == (5,)
    with pytest.raises(ValueError, match="shapes differ"):
        logit_kl(p, q[:, :8])


def test_mlx_helpers_pending_ops_watch_load_and_lazyclock(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from learnai.kit import LazyClock, pending_ops, watch_load

    lazy = mx.ones(4) * 2 + 1
    assert pending_ops(lazy) > 0 and pending_ops([lazy], {"k": lazy}) > 0
    mx.eval(lazy)
    assert pending_ops(lazy) == 0 and pending_ops() == 0 and pending_ops("text", 3) == 0

    path = str(tmp_path / "weights.safetensors")
    mx.save_safetensors(path, {"used": mx.ones(3), "unused": mx.zeros(3)})
    original = mx.load
    with watch_load() as loaded:
        tensors = mx.load(path)
        mx.eval(tensors["used"])
    assert mx.load is original
    assert set(loaded) == {"used", "unused"}
    assert pending_ops(loaded["used"]) == 0 and pending_ops(loaded["unused"]) > 0

    clock = LazyClock(slow=[3.0])
    first, second = mx.ones(2) * 2, mx.ones(2) * 3
    clock.spend(0.5)
    clock.owe(first, 1.0, group="run 0")
    clock.owe(second, 1.0, group="run 1")
    assert clock() == 0.5  # nothing was computed yet
    mx.eval(second)
    assert clock() == 0.5 + 3.0  # the first group collected is the slow one
    mx.eval(first)
    assert clock() == 0.5 + 3.0 + 1.0 and clock.reads == 3
    # Within a group, a later debt waits for the earlier ones.
    clock = LazyClock()
    a, b = mx.ones(2) * 2, mx.ones(2) * 3
    clock.owe(a, 1.0, group="g")
    clock.owe(b, 2.0, group="g")
    mx.eval(b)
    assert clock() == 0.0
    mx.eval(a)
    assert clock() == 3.0


def test_the_kit_imports_without_loading_a_framework():
    code = "import sys, learnai.kit; print([m for m in ('torch', 'jax') if m in sys.modules])"
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code], capture_output=True, text=True, check=False, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


def test_linux_runs_need_no_macos_only_dependency():
    """pyproject: MLX is behind an Apple-silicon marker, so every extra resolves on Linux."""
    import tomllib

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    apple = "sys_platform == 'darwin' and platform_machine == 'arm64'"
    for group in extras.values():
        for requirement in group:
            if requirement.startswith("mlx"):
                assert requirement.endswith(apple), requirement
    assert not any(d.startswith("mlx") for d in project["dependencies"])
