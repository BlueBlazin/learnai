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
```

## Layout

```
src/whetstone/
  harness.py      python -m whetstone.harness: supervisor, watchdog, always emits `end`
  _worker.py      loads the problem, imports the solution, runs the cases
  capture.py      file-descriptor-level output capture
  protocol.py     the @@WS@@ event format
  problem.py      problem.toml / track.toml loading, catalog discovery
  judge/          the test authoring API: @case, Context, check, adapters,
                  speed, simclock, service
  kit/            objects problems hand to solutions: TinyLM, SimModel, Counters
  cli.py          ws list | run | submit | validate | new
  validate.py     the quality gate
  scaffold.py     ws new
  render.py       terminal rendering of the event stream
tests/            pytest suite; fixtures/catalog holds one miniature problem per judge type
```
