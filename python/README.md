# learnai (Python)

The Learn AI judge, kit, and `learnai` CLI. Contracts: `../docs/ARCHITECTURE.md` (sections
3.5 to 3.7). Writing problems: `../docs/PROBLEM_AUTHORING.md`.

Always pass the full set of enabled extras; a bare `uv sync` removes the optional
frameworks.

```sh
uv sync --extra jax --extra mlx
uv run --extra jax --extra mlx pytest
uv run --extra jax --extra mlx ruff check
uv run --extra jax --extra mlx learnai list
uv run --extra jax --extra mlx learnai validate --all
uv run --extra jax --extra mlx learnai doctor --json     # versions and hardware of this environment
```

The same commands work on Linux (x86_64 with CUDA, aarch64): MLX is behind an
Apple-silicon marker, so `--extra mlx` installs nothing there, and torch comes with
its CUDA runtime and Triton.

The `models` extra (huggingface_hub, safetensors, tokenizers, transformers, mlx-lm) is
for real-model problems: `learnai assets fetch` needs it to download weights into
`../.learnai/models/`, and authors need it to generate reference fixtures. The
judge, the kit loaders, and the app do not: they read the downloaded files with numpy.

```sh
uv run --extra jax --extra mlx --extra models learnai assets fetch granite-4.0-350m
uv run --extra jax --extra mlx --extra models pytest   # also compares with transformers
```

The `editor` extra is the ty language server (pinned exactly) that the app's editor
uses for completion, hover, and diagnostics. Nothing in the judge needs it, but once it
is installed, pass `--extra editor` with the others so no command removes it.
`learnai/editor/docs.py` is the app's runtime docs helper (`ARCHITECTURE.md` section 4.4).

## Layout

```
src/learnai/
  harness.py      python -m learnai.harness: supervisor, watchdog, always emits `end`
  _worker.py      loads the problem, imports the solution, runs the cases
  capture.py      file-descriptor-level output capture
  protocol.py     the @@WS@@ event format
  problem.py      problem.toml / track.toml loading, catalog discovery
  assets.py       [[assets]]: declarations, what is on disk, learnai assets list | fetch
  hardware.py     which `requires` this machine meets (mlx-gpu, cuda), and its devices
  judge/          the test authoring API: @case (with stages), Context, check, adapters,
                  speed, simclock, service; warmup.py keeps one-time library costs
                  off a case's clock
  kit/            objects problems hand to solutions: TinyLM, SimModel, Counters,
                  MLXTinyLM and the MLX instruments, RealModel and the real-model loaders,
                  the CUDA-on-CPU tools (cudatools.py) and tritonshim.py, and the
                  milestone 5 set: LayeredLM, TinyMoE, RecurrentLM, TinyEncoder,
                  Grammar, ScriptedChatModel, SimDevices, SimCluster, vectors
  cli.py          learnai list | run | submit | validate | new | assets | doctor
  validate.py     the quality gate
  scaffold.py     learnai new
  render.py       terminal rendering of the event stream
tests/            pytest suite; fixtures/catalog holds one miniature problem per judge type,
                  fixtures/m3 the MLX-only, hardware-requirement, and asset miniatures
```
