# whetstone (Python)

The Whetstone judge, kit, and `ws` CLI. Contracts: `../docs/ARCHITECTURE.md` (sections
3.5 to 3.7). Writing problems: `../docs/PROBLEM_AUTHORING.md`.

Always pass the full set of enabled extras; a bare `uv sync` removes the optional
frameworks.

```sh
uv sync --extra jax --extra mlx
uv run --extra jax --extra mlx pytest
uv run --extra jax --extra mlx ruff check
uv run --extra jax --extra mlx ws list
uv run --extra jax --extra mlx ws validate --all
uv run --extra jax --extra mlx ws doctor --json     # versions and hardware of this environment
```

The same commands work on Linux (x86_64 with CUDA, aarch64): MLX is behind an
Apple-silicon marker, so `--extra mlx` installs nothing there, and torch comes with
its CUDA runtime and Triton.

The `models` extra (huggingface_hub, safetensors, tokenizers, transformers, mlx-lm) is
for real-model problems: `ws assets fetch` needs it to download weights into
`../.whetstone/models/`, and authors need it to generate reference fixtures. The
judge, the kit loaders, and the app do not: they read the downloaded files with numpy.

```sh
uv run --extra jax --extra mlx --extra models ws assets fetch granite-4.0-350m
uv run --extra jax --extra mlx --extra models pytest   # also compares with transformers
```

## Layout

```
src/whetstone/
  harness.py      python -m whetstone.harness: supervisor, watchdog, always emits `end`
  _worker.py      loads the problem, imports the solution, runs the cases
  capture.py      file-descriptor-level output capture
  protocol.py     the @@WS@@ event format
  problem.py      problem.toml / track.toml loading, catalog discovery
  assets.py       [[assets]]: declarations, what is on disk, ws assets list | fetch
  hardware.py     which `requires` this machine meets (mlx-gpu, cuda), and its devices
  judge/          the test authoring API: @case, Context, check, adapters,
                  speed, simclock, service
  kit/            objects problems hand to solutions: TinyLM, SimModel, Counters,
                  MLXTinyLM and the MLX instruments, RealModel and the real-model loaders,
                  the CUDA-on-CPU tools (cudatools.py) and tritonshim.py
  cli.py          ws list | run | submit | validate | new | assets | doctor
  validate.py     the quality gate
  scaffold.py     ws new
  render.py       terminal rendering of the event stream
tests/            pytest suite; fixtures/catalog holds one miniature problem per judge type,
                  fixtures/m3 the MLX-only, hardware-requirement, and asset miniatures
```
