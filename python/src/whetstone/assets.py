"""Model assets: the real open-model weights that some problems need.

A problem declares what it needs in ``problem.toml``::

    [[assets]]
    name = "qwen3-0.6b"                       # the directory under .whetstone/models/
    repo = "Qwen/Qwen3-0.6B"                  # Hugging Face repository
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"   # full commit hash, never a branch
    files = ["config.json", "tokenizer.json", "model.safetensors"]   # names or globs
    size_mb = 1450                            # approximate download size

Weights are never committed. ``ws assets fetch <name>`` downloads them with
``huggingface_hub`` (the ``models`` extra) into ``<root>/.whetstone/models/<name>/``,
which is gitignored, and writes a manifest there. Everything else in this module needs
only the standard library, so the harness, ``ws assets list``, and the kit loaders work
without the extra once the files are on disk.

An asset is *present* when its manifest records the declared repository and revision,
covers every declared file pattern, and every recorded file exists with its recorded
size. The same ``name`` may be declared by many problems; they must agree on ``repo``
and ``revision``, and ``fetch`` downloads the union of their ``files``.

Only safetensors weights and tokenizer or configuration files can be assets
(``is_allowed_file``): nothing that executes or unpickles.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST = "whetstone-asset.json"
NAME_PATTERN = re.compile(r"^[a-z0-9]+([._-][a-z0-9]+)*$")
REPO_PATTERN = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
#: Besides ``*.safetensors``: the small text files a tokenizer and a config consist of.
ALLOWED_NAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
#: Progress events are throttled to one per file per this many seconds.
PROGRESS_INTERVAL_S = 0.25


class AssetError(Exception):
    """An asset declaration is invalid, or a fetch failed."""


class AssetMissing(AssetError):
    """An asset (or one of its files) is not on disk. The message names the fetch command."""


@dataclass(frozen=True)
class AssetSpec:
    """One ``[[assets]]`` entry (or the merge of every entry with the same name)."""

    name: str
    repo: str
    revision: str
    files: tuple[str, ...]
    size_mb: float

    @property
    def fetch_command(self) -> str:
        return f"ws assets fetch {self.name}"

    def describe(self) -> str:
        """`name` (repo, about N MB), for messages."""
        return f"`{self.name}` ({self.repo}, about {format_size(self.size_mb * 1e6)})"


@dataclass(frozen=True)
class AssetState:
    """Where an asset stands on this machine."""

    spec: AssetSpec
    path: Path
    #: ``present`` | ``missing`` (nothing usable on disk) | ``incomplete`` (a directory
    #: exists but it is another revision, lacks declared files, or a file is damaged).
    state: str
    bytes_on_disk: int
    reason: str | None = None

    @property
    def present(self) -> bool:
        return self.state == "present"


def format_size(n_bytes: float) -> str:
    """Decimal units, as Hugging Face and macOS show sizes: 704.8 MB, 3.89 GB."""
    if n_bytes >= 1e9:
        return f"{n_bytes / 1e9:.2f} GB"
    if n_bytes >= 1e6:
        return f"{n_bytes / 1e6:.0f} MB"
    return f"{max(n_bytes, 0) / 1e3:.0f} kB"


def is_allowed_file(path: str) -> bool:
    """Whether a repository file may be downloaded as an asset."""
    name = path.rsplit("/", 1)[-1]
    return name.endswith(".safetensors") or name in ALLOWED_NAMES


def _pattern_can_match_allowed(pattern: str) -> bool:
    name = pattern.rsplit("/", 1)[-1]
    if name.endswith(".safetensors"):
        return True
    return any(fnmatch.fnmatchcase(allowed, name) for allowed in ALLOWED_NAMES)


def parse_assets(raw: Any) -> tuple[list[AssetSpec], list[str]]:
    """Read the ``assets`` value of a problem.toml. Returns ``(specs, problems found)``.

    Lenient like ``load_problem``: valid entries are returned even when others are
    malformed; ``ws validate`` reports the problems.
    """
    if raw is None:
        return [], []
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        return [], ["`assets` must be an array of tables (`[[assets]]`)"]
    specs: list[AssetSpec] = []
    errors: list[str] = []
    for index, item in enumerate(raw):
        where = f"assets[{index}]"
        name, repo, revision = item.get("name"), item.get("repo"), item.get("revision")
        files, size = item.get("files"), item.get("size_mb")
        problems = []
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            problems.append("`name` must be lowercase letters, digits, and . _ - separators")
        else:
            where = f"asset `{name}`"
        if not isinstance(repo, str) or not REPO_PATTERN.match(repo):
            problems.append("`repo` must be a Hugging Face repository id such as `org/model`")
        if not isinstance(revision, str) or not REVISION_PATTERN.match(revision):
            problems.append(
                "`revision` must be a full 40-character commit hash (a branch name can move)"
            )
        if (
            not isinstance(files, list)
            or not files
            or not all(isinstance(f, str) and f for f in files)
        ):
            problems.append("`files` must be a non-empty list of file names or glob patterns")
        else:
            for pattern in files:
                if pattern.startswith(("/", "..")) or "/../" in pattern:
                    problems.append(f"file pattern {pattern!r} must stay inside the repository")
                elif not _pattern_can_match_allowed(pattern):
                    problems.append(
                        f"file pattern {pattern!r} matches no allowed file: assets are "
                        "`*.safetensors` and tokenizer or config files "
                        f"({', '.join(sorted(ALLOWED_NAMES))})"
                    )
        if isinstance(size, bool) or not isinstance(size, (int, float)) or size <= 0:
            problems.append("`size_mb` must be a positive number (the approximate download size)")
        unknown = sorted(set(item) - {"name", "repo", "revision", "files", "size_mb"})
        if unknown:
            problems.append(f"unknown key{'s' if len(unknown) > 1 else ''} {', '.join(unknown)}")
        if any(spec.name == name for spec in specs):
            problems.append("is declared twice")
        errors.extend(f"{where}: {problem}" for problem in problems)
        if not problems:
            specs.append(AssetSpec(name, repo, revision, tuple(files), float(size)))
    return specs, errors


def models_dir(root: Path) -> Path:
    """Where assets live: ``WHETSTONE_MODELS_DIR``, else ``<root>/.whetstone/models``."""
    override = os.environ.get("WHETSTONE_MODELS_DIR")
    return Path(override).resolve() if override else root / ".whetstone" / "models"


def root_of(problem_dir: Path) -> Path:
    """The content root of a problem at ``<root>/problems/<track>/<id>``."""
    env = os.environ.get("WHETSTONE_ROOT")
    return Path(env).resolve() if env else problem_dir.resolve().parents[2]


# -- what is on disk -------------------------------------------------------------------


def read_manifest(directory: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return manifest if isinstance(manifest, dict) else None


def _disk_bytes(directory: Path, manifest: dict[str, Any] | None) -> int:
    total = 0
    for name in (manifest or {}).get("files", {}):
        try:
            total += (directory / name).stat().st_size
        except OSError:
            continue
    return total


def state_of(spec: AssetSpec, directory: Path) -> AssetState:
    """Compare what is in ``directory`` with what ``spec`` declares. No network."""
    path = directory / spec.name
    manifest = read_manifest(path)
    if manifest is None:
        state = "incomplete" if path.is_dir() and any(path.iterdir()) else "missing"
        reason = "a download was started but did not finish" if state == "incomplete" else None
        return AssetState(spec, path, state, 0, reason)
    on_disk = _disk_bytes(path, manifest)

    def incomplete(reason: str) -> AssetState:
        return AssetState(spec, path, "incomplete", on_disk, reason)

    if manifest.get("repo") != spec.repo or manifest.get("revision") != spec.revision:
        return incomplete(
            f"on disk is {manifest.get('repo')} at {str(manifest.get('revision'))[:12]}, "
            f"the catalog pins {spec.repo} at {spec.revision[:12]}"
        )
    covered = manifest.get("patterns") or []
    lacking = [pattern for pattern in spec.files if pattern not in covered]
    if lacking:
        return incomplete(f"not downloaded yet: {', '.join(lacking)}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return incomplete("the manifest lists no files")
    for name, info in files.items():
        try:
            size = (path / name).stat().st_size
        except OSError:
            return incomplete(f"{name} is gone")
        if size != info.get("size"):
            return incomplete(f"{name} has the wrong size")
    return AssetState(spec, path, "present", on_disk)


def missing_message(spec: AssetSpec) -> str:
    return f"model asset {spec.describe()} is not downloaded; fetch it with `{spec.fetch_command}`"


def resolve(spec: AssetSpec, directory: Path, file: str | None = None) -> Path:
    """The directory of a present asset, or the path of one of its files.

    Raises ``AssetMissing`` (naming the fetch command) when the asset is not present
    or the file is not part of it.
    """
    state = state_of(spec, directory)
    if not state.present:
        raise AssetMissing(missing_message(spec))
    if file is None:
        return state.path
    path = state.path / file
    if not path.is_file():
        have = ", ".join(sorted((read_manifest(state.path) or {}).get("files", {})))
        raise AssetMissing(f"asset `{spec.name}` has no file {file!r} (it has: {have})")
    return path


# -- the catalog's view ----------------------------------------------------------------


@dataclass
class CatalogAsset:
    """Every declaration of one asset name across the catalog, merged."""

    spec: AssetSpec
    problems: list[str] = field(default_factory=list)


def merge(
    declarations: Iterable[tuple[str, AssetSpec]],
) -> tuple[dict[str, CatalogAsset], list[tuple[str, str]]]:
    """Merge ``(problem id, spec)`` pairs by asset name.

    Returns the merged assets and ``(problem id, message)`` for every declaration that
    disagrees with an earlier one about the repository or the revision.
    """
    merged: dict[str, CatalogAsset] = {}
    issues: list[tuple[str, str]] = []
    for problem_id, spec in declarations:
        entry = merged.get(spec.name)
        if entry is None:
            merged[spec.name] = CatalogAsset(spec, [problem_id])
            continue
        first = entry.spec
        if (spec.repo, spec.revision) != (first.repo, first.revision):
            issues.append(
                (
                    problem_id,
                    (
                        f"asset `{spec.name}` is {spec.repo} at {spec.revision[:12]} here "
                        f"but {first.repo} at {first.revision[:12]} in {entry.problems[0]}; "
                        "one name must mean one repository and revision"
                    ),
                )
            )
            continue
        files = tuple(dict.fromkeys((*first.files, *spec.files)))
        size = max(first.size_mb, spec.size_mb)
        entry.spec = AssetSpec(first.name, first.repo, first.revision, files, size)
        entry.problems.append(problem_id)
    return merged, issues


def catalog(root: Path) -> tuple[dict[str, CatalogAsset], list[tuple[str, str]]]:
    """Every asset the catalog under ``root`` declares, and the declaration problems."""
    from whetstone.problem import ProblemError, load_problem, problem_dirs

    declarations: list[tuple[str, AssetSpec]] = []
    issues: list[tuple[str, str]] = []
    for directory in problem_dirs(root):
        try:
            problem = load_problem(directory)
        except ProblemError as exc:
            issues.append((directory.name, str(exc)))
            continue
        issues.extend((problem.id, error) for error in problem.asset_errors)
        declarations.extend((problem.id, spec) for spec in problem.assets)
    merged, conflicts = merge(declarations)
    return merged, issues + conflicts


def listing(root: Path) -> dict[str, Any]:
    """The machine-readable form of ``ws assets list`` (see docs/PROBLEM_AUTHORING.md)."""
    merged, issues = catalog(root)
    directory = models_dir(root)
    assets = []
    for name in sorted(merged):
        entry = merged[name]
        state = state_of(entry.spec, directory)
        assets.append(
            {
                "name": name,
                "repo": entry.spec.repo,
                "revision": entry.spec.revision,
                "files": list(entry.spec.files),
                "size_mb": entry.spec.size_mb,
                "state": state.state,
                "present": state.present,
                "reason": state.reason,
                "path": str(state.path),
                "bytes_on_disk": state.bytes_on_disk,
                "problems": sorted(entry.problems),
                "fetch_command": entry.spec.fetch_command,
            }
        )
    return {
        "models_dir": str(directory),
        "assets": assets,
        "issues": [{"problem": problem, "message": message} for problem, message in issues],
    }


# -- fetching ----------------------------------------------------------------------------

Emit = Callable[[dict[str, Any]], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def select_files(siblings: Iterable[tuple[str, int, str | None]], spec: AssetSpec) -> list:
    """The repository files ``spec`` asks for: ``(path, size, sha256 or None)``, sorted.

    A file is taken when it matches one of the patterns *and* is an allowed kind; a
    pattern that selects nothing is an error (a typo must not pass silently).
    """
    siblings = sorted(siblings)
    chosen: dict[str, tuple[str, int, str | None]] = {}
    for pattern in spec.files:
        hits = [s for s in siblings if fnmatch.fnmatchcase(s[0], pattern) and is_allowed_file(s[0])]
        if not hits:
            raise AssetError(
                f"asset `{spec.name}`: file pattern {pattern!r} matches no allowed file in "
                f"{spec.repo} at {spec.revision[:12]}"
            )
        chosen.update({hit[0]: hit for hit in hits})
    return [chosen[path] for path in sorted(chosen)]


def _progress_class(report: Callable[[int, int | None], None]) -> type:
    """A tqdm class that forwards byte counts to ``report`` and draws nothing.

    huggingface_hub shows two bars per file when it downloads through Xet storage:
    "downloading bytes" advances smoothly but counts compressed, deduplicated bytes, so
    it stops short of the file size; "reconstructing file" counts file bytes but moves
    in a few large jumps. Both are forwarded; ``fetch`` reports the larger of the two,
    which is smooth, never goes backwards, and ends exactly at the file size.
    """
    from tqdm import tqdm

    class Progress(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.pop("name", None)  # huggingface_hub's own tqdm accepts it; plain tqdm not
            kwargs["disable"] = False
            kwargs["file"] = open(os.devnull, "w")  # noqa: SIM115 - closed in close()
            self._sink = kwargs["file"]
            super().__init__(*args, **kwargs)
            if self.n:
                report(int(self.n), self.total)

        def update(self, n: float | None = 1) -> bool | None:
            out = super().update(n)
            report(int(self.n), self.total)
            return out

        def close(self) -> None:
            super().close()
            if not self._sink.closed:
                self._sink.close()

    return Progress


def fetch(spec: AssetSpec, directory: Path, emit: Emit, *, verify: bool = True) -> Path:
    """Download ``spec`` into ``directory / spec.name`` and write its manifest.

    Events go to ``emit`` (see docs/PROBLEM_AUTHORING.md for the schema): one
    ``asset_start``, ``asset_progress`` while bytes arrive, one ``asset_file`` per
    file, then ``asset_end``. Files that are already complete are not downloaded again,
    so an interrupted fetch picks up where it stopped, and fetching an asset that is
    present does nothing (and needs neither the network nor ``huggingface_hub``).
    Raises ``AssetError`` after emitting a failed ``asset_end``.
    """
    target = directory / spec.name

    def fail(message: str) -> AssetError:
        emit(
            {
                "event": "asset_end",
                "asset": spec.name,
                "ok": False,
                "path": str(target),
                "bytes": 0,
                "message": message,
            }
        )
        return AssetError(message)

    if state_of(spec, directory).present:
        # Already complete: say so from the manifest, without touching the network.
        recorded = (read_manifest(target) or {}).get("files", {})
        files = [(path, int(info["size"])) for path, info in sorted(recorded.items())]
        total = sum(size for _, size in files)
        emit(
            {
                "event": "asset_start",
                "asset": spec.name,
                "repo": spec.repo,
                "revision": spec.revision,
                "path": str(target),
                "files": [{"path": path, "size": size} for path, size in files],
                "total_bytes": total,
            }
        )
        for path, size in files:
            emit(
                {"event": "asset_file", "asset": spec.name, "file": path, "size": size,
                 "cached": True}
            )  # fmt: skip
        emit(
            {
                "event": "asset_end",
                "asset": spec.name,
                "ok": True,
                "path": str(target),
                "bytes": total,
                "message": None,
            }
        )
        return target

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise fail(
            "downloading needs the `models` extra (huggingface_hub): run this command "
            "with `uv run --project python --extra models ws assets fetch ...`"
        ) from None

    old = read_manifest(target) or {}
    if old and (old.get("repo"), old.get("revision")) != (spec.repo, spec.revision):
        # Another revision's files must not be mistaken for this one's.
        old = {}
    try:
        info = HfApi().model_info(spec.repo, revision=spec.revision, files_metadata=True)
        siblings = [
            (s.rfilename, int(s.size or 0), (getattr(s, "lfs", None) or {}).get("sha256"))
            for s in info.siblings or []
        ]
        files = select_files(siblings, spec)
    except AssetError as exc:
        raise fail(str(exc)) from None
    except Exception as exc:  # noqa: BLE001 - network, authentication, unknown repository
        raise fail(f"cannot read {spec.repo} at {spec.revision[:12]}: {_one_line(exc)}") from None

    total = sum(size for _, size, _ in files)
    emit(
        {
            "event": "asset_start",
            "asset": spec.name,
            "repo": spec.repo,
            "revision": spec.revision,
            "path": str(target),
            "files": [{"path": path, "size": size} for path, size, _ in files],
            "total_bytes": total,
        }
    )
    target.mkdir(parents=True, exist_ok=True)
    recorded: dict[str, dict[str, Any]] = {}
    done = 0
    for path, size, sha256 in files:
        local = target / path
        known = (old.get("files") or {}).get(path)
        cached = (
            known is not None
            and local.is_file()
            and local.stat().st_size == size
            and known.get("size") == size
            and known.get("sha256") == sha256
        )
        if not cached:
            last = [0.0, 0]  # when the last event went out, and the most bytes seen

            def report(
                n: int, _total: int | None, path=path, size=size, last=last, done=done
            ) -> None:
                n = last[1] = max(last[1], min(n, size))
                now = time.monotonic()
                if now - last[0] < PROGRESS_INTERVAL_S and n < size:
                    return
                last[0] = now
                emit(
                    {
                        "event": "asset_progress",
                        "asset": spec.name,
                        "file": path,
                        "file_bytes": min(n, size),
                        "file_total": size,
                        "done_bytes": done + min(n, size),
                        "total_bytes": total,
                    }
                )

            try:
                hf_hub_download(
                    spec.repo,
                    path,
                    revision=spec.revision,
                    local_dir=target,
                    tqdm_class=_progress_class(report),
                )
                if local.stat().st_size != size:
                    raise AssetError(f"{path}: expected {size} bytes, got {local.stat().st_size}")
                if verify and sha256 is not None and _sha256(local) != sha256:
                    local.unlink()
                    raise AssetError(f"{path}: the SHA-256 does not match the repository's")
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - network errors of many kinds
                raise fail(
                    f"downloading {path} failed: {_one_line(exc)}; run the command again to resume"
                ) from None
        done += size
        recorded[path] = {"size": size, "sha256": sha256}
        emit(
            {"event": "asset_file", "asset": spec.name, "file": path, "size": size,
             "cached": cached}
        )  # fmt: skip
        # Written after every file: an interrupted fetch keeps what it finished.
        _write_manifest(target, spec, recorded, complete=False)
    _write_manifest(target, spec, recorded, complete=True)
    emit(
        {
            "event": "asset_end",
            "asset": spec.name,
            "ok": True,
            "path": str(target),
            "bytes": total,
            "message": None,
        }
    )
    return target


def _write_manifest(
    target: Path, spec: AssetSpec, files: dict[str, Any], *, complete: bool
) -> None:
    manifest = {
        "name": spec.name,
        "repo": spec.repo,
        "revision": spec.revision,
        # Patterns count as covered only once every file they select has arrived.
        "patterns": list(spec.files) if complete else [],
        "files": files,
    }
    temporary = target / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target / MANIFEST)


def _one_line(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= 300 else text[:297] + "..."
