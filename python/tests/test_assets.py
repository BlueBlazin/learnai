"""Model assets: declarations, what is on disk, listing, and fetching (no network)."""

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
from conftest import FIXTURES

from whetstone import assets, cli, protocol
from whetstone.problem import load_problem

M3 = FIXTURES / "m3"
REVISION = "0123456789abcdef0123456789abcdef01234567"
GOOD = {
    "name": "tiny",
    "repo": "org/tiny",
    "revision": REVISION,
    "files": ["config.json", "*.safetensors"],
    "size_mb": 1.5,
}


def spec(**changes) -> assets.AssetSpec:
    (parsed,), errors = assets.parse_assets([{**GOOD, **changes}])
    assert errors == []
    return parsed


# -- declarations -------------------------------------------------------------------------


def test_parse_reads_a_valid_table():
    parsed = spec()
    assert parsed == assets.AssetSpec(
        "tiny", "org/tiny", REVISION, ("config.json", "*.safetensors"), 1.5
    )
    assert parsed.fetch_command == "ws assets fetch tiny"
    assert parsed.describe() == "`tiny` (org/tiny, about 2 MB)"
    assert assets.parse_assets(None) == ([], [])


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"name": "Tiny Model"}, "`name` must be lowercase"),
        ({"repo": "tiny"}, "`repo` must be a Hugging Face repository id"),
        ({"revision": "main"}, "`revision` must be a full 40-character commit hash"),
        ({"files": []}, "`files` must be a non-empty list"),
        ({"files": ["pytorch_model.bin"]}, "matches no allowed file"),
        ({"files": ["*.py"]}, "matches no allowed file"),
        ({"files": ["../secrets.json"]}, "must stay inside the repository"),
        ({"size_mb": 0}, "`size_mb` must be a positive number"),
        ({"sha": "abc"}, "unknown key sha"),
    ],
)
def test_parse_reports_what_is_wrong(change, problem):
    specs, errors = assets.parse_assets([{**GOOD, **change}])
    assert specs == [] and len(errors) == 1 and problem in errors[0]


def test_parse_rejects_other_shapes_and_duplicates():
    assert assets.parse_assets(["tiny"])[1] == [
        "`assets` must be an array of tables (`[[assets]]`)"
    ]
    specs, errors = assets.parse_assets([GOOD, GOOD])
    assert len(specs) == 1 and errors == ["asset `tiny`: is declared twice"]


def test_only_safetensors_and_tokenizer_files_are_allowed():
    allowed = ["model.safetensors", "model-00001-of-00002.safetensors", "tokenizer.json",
               "config.json", "merges.txt", "chat_template.jinja", "sub/model.safetensors"]  # fmt: skip
    refused = ["pytorch_model.bin", "model.pkl", "modeling_custom.py", "README.md", "model.gguf"]
    assert all(assets.is_allowed_file(name) for name in allowed)
    assert not any(assets.is_allowed_file(name) for name in refused)


def test_merge_unions_files_and_reports_conflicts():
    a = spec(files=["config.json"], size_mb=1)
    b = spec(files=["*.safetensors", "config.json"], size_mb=700)
    other = spec(revision="f" * 40)
    merged, issues = assets.merge([("p1", a), ("p2", b), ("p3", other)])
    entry = merged["tiny"]
    assert entry.spec.files == ("config.json", "*.safetensors") and entry.spec.size_mb == 700
    assert entry.problems == ["p1", "p2"]
    ((problem, message),) = issues
    assert problem == "p3" and "one name must mean one repository and revision" in message


def test_select_files_filters_and_insists_that_patterns_match():
    siblings = [("config.json", 10, None), ("model.safetensors", 99, "abc"), ("evil.py", 5, None),
                ("pytorch_model.bin", 99, None)]  # fmt: skip
    chosen = assets.select_files(siblings, spec(files=["*.safetensors", "config.json"]))
    assert chosen == [("config.json", 10, None), ("model.safetensors", 99, "abc")]
    # A glob wide enough to match code still only takes allowed files.
    assert assets.select_files(siblings, spec(files=["*.json"])) == [("config.json", 10, None)]
    with pytest.raises(assets.AssetError, match="'tokenizer.json' matches no allowed file"):
        assets.select_files(siblings, spec(files=["tokenizer.json"]))


def test_models_dir_and_format_size(tmp_path, monkeypatch):
    monkeypatch.delenv("WHETSTONE_MODELS_DIR", raising=False)
    assert assets.models_dir(tmp_path) == tmp_path / ".whetstone" / "models"
    monkeypatch.setenv("WHETSTONE_MODELS_DIR", str(tmp_path / "elsewhere"))
    assert assets.models_dir(tmp_path) == (tmp_path / "elsewhere").resolve()
    assert [assets.format_size(n) for n in (704_786_224, 3_888_771_048, 10_000)] == [
        "705 MB", "3.89 GB", "10 kB",
    ]  # fmt: skip


# -- fetching, against a stand-in for huggingface_hub ---------------------------------------


class FakeHub:
    """Just enough of huggingface_hub: a repository of bytes, and a download log."""

    def __init__(self, files: dict[str, bytes], *, corrupt: str | None = None) -> None:
        self.files = files
        self.corrupt = corrupt
        self.downloads: list[str] = []
        self.revisions: list[str] = []

    def install(self, monkeypatch) -> None:
        # Progress goes through a tqdm class, which arrives with huggingface_hub.
        pytest.importorskip("tqdm", reason="needs the `models` extra")
        hub = self
        module = types.ModuleType("huggingface_hub")

        class HfApi:
            def model_info(self, repo, *, revision, files_metadata):
                hub.revisions.append(revision)
                siblings = [
                    types.SimpleNamespace(
                        rfilename=name,
                        size=len(data),
                        lfs={"sha256": hashlib.sha256(data).hexdigest()}
                        if name.endswith(".safetensors")
                        else None,
                    )
                    for name, data in hub.files.items()
                ]
                return types.SimpleNamespace(siblings=siblings)

        def hf_hub_download(repo, filename, *, revision, local_dir, tqdm_class):
            hub.downloads.append(filename)
            data = hub.files[filename]
            if filename == hub.corrupt:
                data = b"X" + data[1:]
            bar = tqdm_class(total=len(data), initial=0, desc=filename, unit="B")
            half = len(data) // 2
            bar.update(half)
            bar.update(len(data) - half)
            bar.close()
            path = Path(local_dir) / filename
            path.write_bytes(data)
            return str(path)

        module.HfApi = HfApi
        module.hf_hub_download = hf_hub_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)


REPO = {
    "config.json": b'{"vocab_size": 11}',
    "model.safetensors": b"\x02\x00\x00\x00\x00\x00\x00\x00{}" + bytes(4096),
    "README.md": b"not an asset",
    "modeling.py": b"print('never downloaded')",
}


def test_fetch_downloads_pinned_allowed_files_and_writes_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "PROGRESS_INTERVAL_S", 0.0)
    hub = FakeHub(REPO)
    hub.install(monkeypatch)
    events: list[dict] = []
    target = assets.fetch(spec(), tmp_path, events.append)

    assert target == tmp_path / "tiny"
    assert hub.revisions == [REVISION]
    assert hub.downloads == ["config.json", "model.safetensors"]
    assert sorted(p.name for p in target.iterdir()) == [
        "config.json", "model.safetensors", assets.MANIFEST,
    ]  # fmt: skip
    total = len(REPO["config.json"]) + len(REPO["model.safetensors"])
    assert events[0] == {
        "event": "asset_start",
        "asset": "tiny",
        "repo": "org/tiny",
        "revision": REVISION,
        "path": str(target),
        "files": [
            {"path": "config.json", "size": len(REPO["config.json"])},
            {"path": "model.safetensors", "size": len(REPO["model.safetensors"])},
        ],
        "total_bytes": total,
    }
    progress = [e for e in events if e["event"] == "asset_progress"]
    assert progress[-1] == {
        "event": "asset_progress",
        "asset": "tiny",
        "file": "model.safetensors",
        "file_bytes": len(REPO["model.safetensors"]),
        "file_total": len(REPO["model.safetensors"]),
        "done_bytes": total,
        "total_bytes": total,
    }
    assert [e["done_bytes"] for e in progress] == sorted(e["done_bytes"] for e in progress)
    assert [(e["file"], e["cached"]) for e in events if e["event"] == "asset_file"] == [
        ("config.json", False), ("model.safetensors", False),
    ]  # fmt: skip
    assert events[-1] == {
        "event": "asset_end", "asset": "tiny", "ok": True, "path": str(target),
        "bytes": total, "message": None,
    }  # fmt: skip
    state = assets.state_of(spec(), tmp_path)
    assert state.present and state.bytes_on_disk == total
    assert assets.resolve(spec(), tmp_path, "config.json").read_bytes() == REPO["config.json"]
    with pytest.raises(assets.AssetMissing, match="has no file 'tokenizer.json'"):
        assets.resolve(spec(), tmp_path, "tokenizer.json")


def test_fetch_again_downloads_nothing_and_resumes_what_is_missing(tmp_path, monkeypatch):
    hub = FakeHub(REPO)
    hub.install(monkeypatch)
    assets.fetch(spec(), tmp_path, lambda event: None)
    hub.downloads.clear()
    hub.revisions.clear()
    events: list[dict] = []
    assets.fetch(spec(), tmp_path, events.append)
    assert hub.downloads == [] and hub.revisions == []  # present: the network is not asked
    assert [e["event"] for e in events] == ["asset_start", "asset_file", "asset_file", "asset_end"]
    assert [e["cached"] for e in events if e["event"] == "asset_file"] == [True, True]
    assert events[-1]["ok"] is True and events[0]["total_bytes"] == events[-1]["bytes"]

    # One file complete, one missing: only the missing one is downloaded.
    (tmp_path / "tiny" / "model.safetensors").unlink()
    events.clear()
    assets.fetch(spec(), tmp_path, events.append)
    assert hub.downloads == ["model.safetensors"]
    assert [(e["file"], e["cached"]) for e in events if e["event"] == "asset_file"] == [
        ("config.json", True), ("model.safetensors", False),
    ]  # fmt: skip
    hub.downloads.clear()

    (tmp_path / "tiny" / "model.safetensors").unlink()  # as if interrupted mid-way
    assert assets.state_of(spec(), tmp_path).state == "incomplete"
    assets.fetch(spec(), tmp_path, lambda event: None)
    assert hub.downloads == ["model.safetensors"]
    assert assets.state_of(spec(), tmp_path).present


def test_fetch_failures_end_with_a_failed_event_not_a_crash(tmp_path, monkeypatch):
    FakeHub(REPO, corrupt="model.safetensors").install(monkeypatch)
    events: list[dict] = []
    with pytest.raises(assets.AssetError, match="SHA-256 does not match"):
        assets.fetch(spec(), tmp_path, events.append)
    assert events[-1]["event"] == "asset_end" and events[-1]["ok"] is False
    assert "run the command again to resume" in events[-1]["message"]
    assert not (tmp_path / "tiny" / "model.safetensors").exists()
    assert assets.state_of(spec(), tmp_path).state == "incomplete"

    events.clear()
    with pytest.raises(assets.AssetError, match="matches no allowed file"):
        assets.fetch(spec(files=["tokenizer.json"]), tmp_path, events.append)
    assert [e["event"] for e in events] == ["asset_end"]


def test_fetch_without_the_models_extra_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import raises ImportError
    events: list[dict] = []
    with pytest.raises(assets.AssetError, match="needs the `models` extra"):
        assets.fetch(spec(), tmp_path, events.append)
    assert events[-1]["ok"] is False and "--extra models" in events[-1]["message"]


# -- the CLI ----------------------------------------------------------------------------------


@pytest.fixture
def m3_root(tmp_path, monkeypatch):
    monkeypatch.setenv("WHETSTONE_ROOT", str(M3))
    monkeypatch.setenv("WHETSTONE_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path / "models"


def test_assets_list_json_is_one_document(m3_root, capsys):
    assert cli.main(["assets", "list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing == {
        "models_dir": str(m3_root),
        "assets": [
            {
                "name": "tiny-test-model",
                "repo": "whetstone-tests/tiny-test-model",
                "revision": REVISION,
                "files": ["config.json", "*.safetensors"],
                "size_mb": 0.01,
                "state": "missing",
                "present": False,
                "reason": None,
                "path": str(m3_root / "tiny-test-model"),
                "bytes_on_disk": 0,
                "problems": ["asset-vocab"],
                "fetch_command": "ws assets fetch tiny-test-model",
            }
        ],
        "issues": [],
    }


def test_assets_fetch_and_list_from_the_command_line(m3_root, capsys, monkeypatch):
    monkeypatch.setattr(assets, "PROGRESS_INTERVAL_S", 0.0)
    FakeHub(REPO).install(monkeypatch)
    assert cli.main(["assets", "list"]) == 0
    out = capsys.readouterr().out
    assert "tiny-test-model  missing  ~10 kB" in out
    assert "tiny-test-model is missing: ws assets fetch tiny-test-model" in out

    assert cli.main(["assets", "fetch", "tiny-test-model", "--json"]) == 0
    lines = capsys.readouterr().out.splitlines()
    events = [protocol.decode(line) for line in lines]
    assert all(event is not None for event in events)  # every line is a protocol line
    kinds = [event["event"] for event in events]
    assert kinds[0] == "asset_start" and kinds[-1] == "asset_end" and "asset_progress" in kinds
    assert events[-1]["ok"] is True

    assert cli.main(["assets", "list", "--json"]) == 0
    (item,) = json.loads(capsys.readouterr().out)["assets"]
    assert item["present"] is True and item["state"] == "present" and item["bytes_on_disk"] > 4000

    assert cli.main(["assets", "fetch", "tiny-test-model"]) == 0
    out = capsys.readouterr().out
    assert "done   config.json (already downloaded)" in out and "ready at" in out


def test_assets_fetch_usage_errors_and_failures(m3_root, capsys, monkeypatch):
    assert cli.main(["assets", "fetch", "no-such-model"]) == 2
    assert "no asset named no-such-model" in capsys.readouterr().err
    assert cli.main(["assets", "fetch"]) == 2
    capsys.readouterr()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert cli.main(["assets", "fetch", "tiny-test-model", "--json"]) == 1
    (event,) = [protocol.decode(line) for line in capsys.readouterr().out.splitlines()]
    assert event["event"] == "asset_end" and event["ok"] is False


def test_house_models_can_be_fetched_before_a_problem_declares_them(m3_root, capsys, monkeypatch):
    from whetstone.kit.realmodel import HOUSE_MODELS, SMALL

    seen: list[assets.AssetSpec] = []
    monkeypatch.setattr(assets, "fetch", lambda spec, *args, **kwargs: seen.append(spec))
    assert cli.main(["assets", "fetch", SMALL.name]) == 0
    assert seen == [SMALL]
    assert set(HOUSE_MODELS) == {"granite-4.0-350m", "granite-4.2-3b-mlx-8bit"}
    for house in HOUSE_MODELS.values():
        (parsed,), errors = assets.parse_assets(
            [
                {
                    "name": house.name,
                    "repo": house.repo,
                    "revision": house.revision,
                    "files": list(house.files),
                    "size_mb": house.size_mb,
                }
            ]
        )
        assert errors == [] and parsed == house


def test_validate_checks_asset_declarations(tmp_path, monkeypatch):
    import shutil

    from whetstone import validate as gate

    root = tmp_path / "root"
    shutil.copytree(M3 / "problems" / "real", root / "problems" / "real")
    monkeypatch.setenv("WHETSTONE_ROOT", str(root))
    monkeypatch.setenv("WHETSTONE_MODELS_DIR", str(tmp_path / "models"))
    problem = root / "problems" / "real" / "asset-vocab"
    toml = problem / "problem.toml"
    original = toml.read_text()

    toml.write_text(original.replace('"*.safetensors"', '"*.bin"'))
    (report,) = gate.validate(root, [problem])
    assert len(report.errors) == 1 and "matches no allowed file" in report.errors[0]

    # A house model's name must mean the house model's repository and revision.
    toml.write_text(original.replace('"tiny-test-model"', '"granite-4.0-350m"'))
    report, _ = gate.check_metadata(problem, {"asset-vocab": [problem]})
    assert any(
        "is a house model and must be ibm-granite/granite-4.0-350m" in e for e in report.errors
    )

    # Two problems, one asset name, two revisions.
    toml.write_text(original)
    twin = root / "problems" / "real" / "asset-twin"
    shutil.copytree(problem, twin)
    (twin / "problem.toml").write_text(
        original.replace('id = "asset-vocab"', 'id = "asset-twin"').replace(REVISION, "f" * 40)
    )
    reports = gate.validate(root, [problem, twin])
    conflicts = [e for r in reports for e in r.errors if "one name must mean" in e]
    assert len(conflicts) == 1
    assert load_problem(twin).assets[0].revision == "f" * 40
