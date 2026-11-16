"""Milestone 3 harness behavior: MLX-only problems, per-case hardware requirements,
model assets, and the verdicts for runs that could not run every case."""

import json
import shutil
from pathlib import Path

import pytest
from conftest import FIXTURES, run_harness

from learnai import assets, cli, hardware, protocol
from learnai import validate as gate
from learnai.problem import load_problem

M3 = FIXTURES / "m3"
MLX_SCALE = M3 / "problems" / "gpu" / "mlx-scale"
CUDA_SCALE = M3 / "problems" / "gpu" / "cuda-scale"
ASSET_VOCAB = M3 / "problems" / "real" / "asset-vocab"

needs_mlx = pytest.mark.skipif(
    not hardware.available("mlx-gpu"), reason="needs MLX with an Apple GPU"
)


def install_asset(models: Path, problem_dir: Path = ASSET_VOCAB, *, vocab_size: int = 11) -> Path:
    """Put the fixture problem's asset on disk the way ``learnai assets fetch`` would."""
    (spec,) = load_problem(problem_dir).assets
    target = models / spec.name
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(json.dumps({"vocab_size": vocab_size}))
    (target / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    files = {
        name: {"size": (target / name).stat().st_size, "sha256": None}
        for name in ("config.json", "model.safetensors")
    }
    assets._write_manifest(target, spec, files, complete=True)
    return target


# -- verdict rules ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("statuses", "kinds", "hardware_met", "verdict"),
    [
        (["pass", "pass"], [], True, "accepted"),
        (["pass", "skipped"], ["hardware"], False, "hardware_missing"),
        (["pass", "skipped"], ["asset"], True, "assets_missing"),
        (["skipped", "skipped"], ["asset", "hardware"], False, "assets_missing"),
        # A problem whose hardware is absent is never accepted, even with no case marked.
        (["pass", "pass"], [], False, "hardware_missing"),
        # What ran decides first.
        (["fail", "skipped"], ["hardware"], False, "wrong_answer"),
        (["error", "skipped"], ["asset"], True, "runtime_error"),
        (["timeout", "fail", "skipped"], ["asset"], True, "time_limit"),
        # Skipped because the run stopped early: as before, never accepted.
        (["pass", "skipped"], [], True, "wrong_answer"),
        (["pass", "skipped", "skipped"], ["hardware"], False, "wrong_answer"),
    ],
)
def test_verdict_rules_for_skipped_cases(statuses, kinds, hardware_met, verdict):
    got = protocol.verdict_for(statuses, skip_kinds=kinds, hardware_met=hardware_met)
    assert got == verdict


def test_run_level_metrics_keep_the_worst_value_in_either_direction():
    cases = [
        [{"name": "slowdown", "value": 1.5}, {"name": "tokens_per_s", "value": 900.0,
                                             "higher_is_better": True}],
        [{"name": "slowdown", "value": 2.5}, {"name": "tokens_per_s", "value": 400.0,
                                             "higher_is_better": True}],
    ]  # fmt: skip
    worst = {m["name"]: m["value"] for m in protocol.summarize_metrics(cases)}
    assert worst == {"slowdown": 2.5, "tokens_per_s": 400.0}


# -- hardware requirements --------------------------------------------------------------


def test_hardware_override_and_descriptions(monkeypatch):
    monkeypatch.setenv("LEARNAI_HARDWARE", "cpu")
    assert hardware.available("cpu")
    assert not hardware.available("mlx-gpu") and not hardware.available("cuda")
    monkeypatch.setenv("LEARNAI_HARDWARE", "cuda")
    assert hardware.available("cuda") and hardware.available("cpu")
    assert "NVIDIA" in hardware.missing_message("cuda")


def test_cuda_problem_without_cuda_runs_cpu_cases_and_skips_the_rest():
    env = {"LEARNAI_HARDWARE": "cpu"}
    run = run_harness(CUDA_SCALE, CUDA_SCALE / "reference" / "torch.py", "torch", env=env)
    assert run.returncode == 0
    assert run.start["requires"] == "cuda" and run.start["requires_met"] is False
    by_id = {c["id"]: c for c in run.start["cases"]}
    assert by_id["cpu"].get("requires") is None
    assert run.statuses == {"cpu": "pass", "hidden-1": "pass", "hidden-2": "skipped"}
    skipped = run.case("hidden-2")
    assert skipped["skip"] == {"kind": "hardware", "requires": "cuda"}
    assert skipped["message"] == (
        "skipped: needs an NVIDIA GPU (CUDA), which this machine does not have"
    )
    assert skipped["detail"] is None
    # A skipped case has no case_start.
    assert [e["id"] for e in run.of("case_start")] == ["cpu", "hidden-1"]
    end = run.end
    assert end["status"] == "hardware_missing"
    assert (end["passed"], end["total"], end["skipped"]) == (2, 3, 1)
    assert end["missing_hardware"] == "cuda"
    assert end["message"].startswith(
        "2 of 2 cases that could run passed; 1 of 3 cases were skipped"
    )


def test_a_failure_outranks_missing_hardware():
    env = {"LEARNAI_HARDWARE": "cpu"}
    run = run_harness(CUDA_SCALE, CUDA_SCALE / "mutants" / "ignores_factor.py", "torch", env=env)
    assert run.end["status"] == "wrong_answer"
    assert run.statuses["hidden-2"] == "skipped"


def test_run_mode_with_only_cpu_cases_still_reports_the_missing_hardware():
    env = {"LEARNAI_HARDWARE": "cpu"}
    run = run_harness(
        CUDA_SCALE, CUDA_SCALE / "reference" / "torch.py", "torch", mode="run", env=env
    )
    assert run.statuses == {"cpu": "pass"}
    assert run.end["status"] == "hardware_missing"
    assert run.end["message"].startswith("every case passed, but this problem requires")


def test_selecting_one_skipped_case_reports_it_without_importing_the_solution(tmp_path):
    crashing = tmp_path / "torch.py"
    crashing.write_text("raise RuntimeError('imported')\n")
    env = {"LEARNAI_HARDWARE": "cpu"}
    run = run_harness(CUDA_SCALE, crashing, "torch", case="hidden-2", env=env)
    assert run.statuses == {"hidden-2": "skipped"}
    assert run.end["status"] == "hardware_missing" and run.end["passed"] == 0


def test_a_case_may_only_require_what_the_problem_requires(tmp_path):
    problem = tmp_path / "problems" / "gpu" / "cuda-scale"
    shutil.copytree(CUDA_SCALE, problem)
    toml = problem / "problem.toml"
    toml.write_text(toml.read_text().replace('requires = "cuda"', 'requires = "cpu"'))
    run = run_harness(problem, problem / "reference" / "torch.py", "torch")
    assert run.end["status"] == "harness_error"
    assert "has requires='cuda', but problem.toml says requires = 'cpu'" in run.end["message"]


def test_case_decorator_rejects_unknown_requirements():
    from learnai.judge import case

    with pytest.raises(ValueError, match="requires= must be one of cpu, mlx-gpu, cuda"):
        case("x", requires="gpu")
    with pytest.raises(ValueError, match="assets= must be a list"):
        case("x", assets="model")


# -- MLX-only problems ------------------------------------------------------------------


def test_the_oracle_of_an_mlx_only_problem_is_its_mlx_reference():
    assert load_problem(MLX_SCALE).oracle_framework == "mlx"
    assert load_problem(CUDA_SCALE).oracle_framework == "torch"
    assert load_problem(ASSET_VOCAB).oracle_framework == "python"


@needs_mlx
def test_mlx_only_problem_runs_with_memory_and_throughput_metrics():
    run = run_harness(MLX_SCALE, MLX_SCALE / "reference" / "mlx.py", "mlx")
    assert run.end["status"] == "accepted", run.events
    assert run.start["requires"] == "mlx-gpu" and run.start["requires_met"] is True
    assert "torch" not in run.start["env"] and "mlx" in run.start["env"]
    assert [c.get("requires") for c in run.start["cases"]] == [None, None, "mlx-gpu", "mlx-gpu"]
    (memory,) = run.case("hidden-2")["metrics"]
    assert memory["name"] == "peak_memory" and memory["unit"] == "MiB" and memory["limit"] == 3
    assert 1.9 <= memory["value"] <= 3.0
    metrics = {m["name"]: m for m in run.case("hidden-3")["metrics"]}
    assert metrics["slowdown"]["limit"] == 20
    assert (
        metrics["tokens_per_s"]["higher_is_better"] is True
        and "limit" not in metrics["tokens_per_s"]
    )
    assert {m["name"] for m in run.end["metrics"]} == {"peak_memory", "slowdown", "tokens_per_s"}


@needs_mlx
def test_the_memory_limit_catches_a_float32_promotion():
    run = run_harness(MLX_SCALE, MLX_SCALE / "mutants" / "promotes.py", "mlx")
    assert run.end["status"] == "wrong_answer"
    failed = run.case("hidden-2")
    assert failed["status"] == "fail" and "float32" in failed["message"]


@needs_mlx
def test_mlx_problem_without_a_gpu_skips_its_gpu_cases():
    env = {"LEARNAI_HARDWARE": "cpu"}
    run = run_harness(MLX_SCALE, MLX_SCALE / "reference" / "mlx.py", "mlx", env=env)
    assert run.statuses == {
        "small": "pass", "hidden-1": "pass", "hidden-2": "skipped", "hidden-3": "skipped",
    }  # fmt: skip
    assert run.end["status"] == "hardware_missing"
    assert "an Apple GPU (MLX with Metal)" in run.end["message"]


@needs_mlx
def test_validate_accepts_an_mlx_only_problem_with_mlx_mutants(monkeypatch):
    monkeypatch.setenv("LEARNAI_ROOT", str(M3))
    (report,) = gate.validate(M3, [MLX_SCALE])
    assert report.errors == [] and report.warnings == [] and report.status == "ok"
    assert "mutant promotes rejected (wrong_answer)" in report.checks
    assert "reference (mlx) accepted, 4 cases" in " ".join(report.checks)


def test_validate_requires_torch_unless_the_problem_is_mlx_only(tmp_path, monkeypatch):
    root = tmp_path / "root"
    shutil.copytree(M3 / "problems" / "gpu", root / "problems" / "gpu")
    monkeypatch.setenv("LEARNAI_ROOT", str(root))
    problem = root / "problems" / "gpu" / "mlx-scale"
    toml = problem / "problem.toml"
    toml.write_text(toml.read_text().replace('["mlx"]', '["jax", "mlx"]'))
    report, _ = gate.check_metadata(problem, {"mlx-scale": [problem]})
    assert any("tensor problems must support" in e and "MLX-only" in e for e in report.errors)
    toml.write_text(
        toml.read_text().replace('["jax", "mlx"]', '["torch"]').replace("mlx-gpu", "mlx-gpu")
    )
    report, _ = gate.check_metadata(problem, {"mlx-scale": [problem]})
    assert 'problem.toml: requires = "mlx-gpu" needs "mlx" in frameworks' in report.errors


def test_new_scaffolds_an_mlx_only_problem(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    (root / "problems").mkdir(parents=True)
    (root / "python").mkdir()
    monkeypatch.setenv("LEARNAI_ROOT", str(root))
    assert cli.main(["new", "mlx/lazy-thing", "--frameworks", "mlx"]) == 0
    problem = load_problem(root / "problems" / "mlx" / "lazy-thing")
    assert problem.frameworks == ("mlx",) and problem.requires == "mlx-gpu"
    assert (problem.path / "reference" / "mlx.py").is_file()
    assert "import mlx.core as mx" in (problem.path / "mutants" / "ignores_factor.py").read_text()
    assert cli.main(["new", "mlx/other", "--frameworks", "jax"]) == 2
    assert "MLX-only" in capsys.readouterr().err


# -- validate on a machine that lacks the hardware ----------------------------------------


def test_validate_reports_cuda_cases_it_could_not_run(monkeypatch, capsys):
    monkeypatch.setenv("LEARNAI_ROOT", str(M3))
    monkeypatch.setenv("LEARNAI_HARDWARE", "cpu")
    (report,) = gate.validate(M3, [CUDA_SCALE])
    assert report.errors == [] and report.status == "ok"
    reference, mutant = report.warnings
    assert reference == (
        "reference (torch): only 2 of 3 cases could run on this machine (all passed); 1 was "
        "NOT validated: needs an NVIDIA GPU (CUDA), which this machine does not have "
        "(GPU cases not validated: on_cuda)"
    )
    # The mutant only a GPU can catch is reported, not silently counted as rejected.
    assert mutant.startswith("mutant stays_on_cpu: NOT validated: no case that can run")
    assert "mutant ignores_factor rejected (wrong_answer)" in report.checks
    assert "starter (torch) rejected (runtime_error)" in report.checks

    assert cli.main(["validate", "cuda-scale"]) == 0
    out = capsys.readouterr().out
    assert "ok   gpu/cuda-scale" in out and "warning: reference (torch): only 2 of 3" in out


def test_validate_fails_a_starter_that_only_missing_hardware_holds_back(tmp_path, monkeypatch):
    root = tmp_path / "root"
    shutil.copytree(M3 / "problems" / "gpu", root / "problems" / "gpu")
    monkeypatch.setenv("LEARNAI_ROOT", str(root))
    monkeypatch.setenv("LEARNAI_HARDWARE", "cpu")
    problem = root / "problems" / "gpu" / "cuda-scale"
    (problem / "starter" / "torch.py").write_text(
        "import torch\n\n\ndef scale(x, factor, device):\n    return x * factor  # same as CPU\n"
    )
    (report,) = gate.validate(root, [problem])
    assert report.errors == [
        (
            "starter (torch): passes every case that can run on this machine; the cases that run "
            "everywhere must reject the starter"
        )
    ]


# -- assets in the harness ----------------------------------------------------------------


def test_missing_asset_skips_the_cases_that_need_it(tmp_path):
    env = {"LEARNAI_MODELS_DIR": str(tmp_path / "models")}
    run = run_harness(ASSET_VOCAB, ASSET_VOCAB / "reference" / "python.py", env=env)
    assert run.returncode == 0
    assert run.statuses == {
        "synthetic": "pass", "hidden-1": "pass", "real_config": "skipped", "hidden-2": "skipped",
    }  # fmt: skip
    skipped = run.case("real_config")
    assert skipped["skip"] == {"kind": "asset", "asset": "tiny-test-model"}
    assert skipped["message"] == (
        "skipped: model asset `tiny-test-model` (learnai-tests/tiny-test-model, about 10 kB) "
        "is not downloaded; fetch it with `learnai assets fetch tiny-test-model`"
    )
    end = run.end
    assert end["status"] == "assets_missing"
    assert (end["passed"], end["total"], end["skipped"]) == (2, 4, 2)
    assert "Fetch with `learnai assets fetch tiny-test-model`, then run again." in end["message"]
    assert end["missing_assets"] == [
        {
            "name": "tiny-test-model",
            "repo": "learnai-tests/tiny-test-model",
            "size_mb": 0.01,
            "fetch_command": "learnai assets fetch tiny-test-model",
        }
    ]


def test_present_asset_is_resolved_for_the_tests(tmp_path):
    models = tmp_path / "models"
    install_asset(models)
    env = {"LEARNAI_MODELS_DIR": str(models)}
    run = run_harness(ASSET_VOCAB, ASSET_VOCAB / "reference" / "python.py", env=env)
    assert run.end["status"] == "accepted", run.events
    assert run.end["skipped"] == 0
    run = run_harness(ASSET_VOCAB, ASSET_VOCAB / "mutants" / "constant.py", env=env)
    assert run.end["status"] == "wrong_answer"


def test_a_stale_or_damaged_asset_counts_as_missing(tmp_path):
    models = tmp_path / "models"
    target = install_asset(models)
    (spec,) = load_problem(ASSET_VOCAB).assets
    assert assets.state_of(spec, models).state == "present"
    (target / "model.safetensors").write_bytes(b"short")
    state = assets.state_of(spec, models)
    assert state.state == "incomplete" and state.reason == "model.safetensors has the wrong size"
    install_asset(models)
    manifest = json.loads((target / assets.MANIFEST).read_text())
    manifest["revision"] = "f" * 40
    (target / assets.MANIFEST).write_text(json.dumps(manifest))
    state = assets.state_of(spec, models)
    assert state.state == "incomplete" and "the catalog pins" in state.reason
    env = {"LEARNAI_MODELS_DIR": str(models)}
    run = run_harness(ASSET_VOCAB, ASSET_VOCAB / "reference" / "python.py", env=env)
    assert run.end["status"] == "assets_missing"


def test_a_malformed_assets_table_is_a_harness_error_not_a_crash(tmp_path):
    problem = tmp_path / "problems" / "real" / "asset-vocab"
    shutil.copytree(ASSET_VOCAB, problem)
    toml = problem / "problem.toml"
    toml.write_text(toml.read_text().replace("0123456789abcdef0123456789abcdef01234567", "main"))
    run = run_harness(problem, problem / "reference" / "python.py")
    assert run.end["status"] == "harness_error"
    assert "`revision` must be a full 40-character commit hash" in run.end["message"]


def test_validate_skips_what_needs_a_missing_asset_and_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LEARNAI_ROOT", str(M3))
    monkeypatch.setenv("LEARNAI_MODELS_DIR", str(tmp_path / "models"))
    (report,) = gate.validate(M3, [ASSET_VOCAB])
    assert report.errors == [] and report.status == "ok"
    assert report.warnings[0].startswith(
        "reference (python): only 2 of 4 cases could run on this machine (all passed); 2 were "
        "NOT validated: model asset `tiny-test-model`"
    )
    assert report.warnings[1].startswith("mutant constant: NOT validated")

    (report,) = gate.validate(M3, [ASSET_VOCAB], require_assets=True)
    assert report.status == "fail"
    assert report.errors[0].startswith("reference (python): a required model asset is missing")
    assert cli.main(["validate", "asset-vocab", "--require-assets"]) == 1
    capsys.readouterr()

    install_asset(tmp_path / "models")
    (report,) = gate.validate(M3, [ASSET_VOCAB], require_assets=True)
    assert report.errors == [] and report.warnings == []
    assert "mutant constant rejected (wrong_answer)" in report.checks


def test_validate_reports_a_problem_of_which_nothing_could_run_as_skipped(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "root"
    shutil.copytree(M3 / "problems" / "real", root / "problems" / "real")
    monkeypatch.setenv("LEARNAI_ROOT", str(root))
    monkeypatch.setenv("LEARNAI_MODELS_DIR", str(tmp_path / "models"))
    problem = root / "problems" / "real" / "asset-vocab"
    tests = problem / "tests.py"
    tests.write_text(tests.read_text().replace(", assets=[]", ""))  # every case needs the asset
    (report,) = gate.validate(root, [problem])
    assert report.status == "skipped" and report.errors == [] and not report.ran
    assert report.skips == [
        (
            "reference (python): not validated here: model asset `tiny-test-model` "
            "(learnai-tests/tiny-test-model, about 10 kB) is not downloaded; fetch it with "
            "`learnai assets fetch tiny-test-model`"
        )
    ]
    assert cli.main(["validate", "asset-vocab"]) == 0
    out = capsys.readouterr().out
    assert "SKIP real/asset-vocab" in out
    assert "0 of 1 problems passed validation; 1 skipped (not validated on this machine" in out
    assert cli.main(["validate", "asset-vocab", "--require-assets"]) == 1


def test_fixtures_are_loaded_by_extension(tmp_path):
    import numpy as np

    from learnai.judge.context import make_context
    from learnai.judge.registry import Case

    problem = tmp_path / "problems" / "real" / "asset-vocab"
    shutil.copytree(ASSET_VOCAB, problem)
    np.savez(problem / "fixtures" / "logits.npz", logits=np.arange(6.0).reshape(2, 3))
    np.save(problem / "fixtures" / "ids.npy", np.array([1, 2, 3]))
    (problem / "fixtures" / "prompt.txt").write_text("hello\n")
    (problem / "fixtures" / "blob.bin").write_bytes(b"\x00\x01")
    case = Case(id="unit", name="unit", public=True, fn=lambda ctx: None)
    ctx = make_context(load_problem(problem), solution=None, framework="python", case=case)
    assert ctx.fixture("expected.json") == {"vocab_size": 11}
    assert ctx.fixture("logits.npz")["logits"].shape == (2, 3)
    assert ctx.fixture("ids.npy").tolist() == [1, 2, 3]
    assert ctx.fixture("prompt.txt") == "hello\n" and ctx.fixture("blob.bin") == b"\x00\x01"
    with pytest.raises(FileNotFoundError, match="no fixture 'missing.npz'"):
        ctx.fixture("missing.npz")
    with pytest.raises(FileNotFoundError):
        ctx.fixture("../problem.toml")
    with pytest.raises(LookupError, match="not declared in `\\[\\[assets\\]\\]`"):
        ctx.asset("some-other-model")
