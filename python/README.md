# whetstone (Python)

The Whetstone judge, kit, and `ws` CLI. See `../docs/ARCHITECTURE.md`.

Always pass the full set of enabled extras; a bare `uv sync` removes the optional frameworks.

```sh
uv sync --extra jax --extra mlx
uv run --extra jax --extra mlx ws --version
uv run --extra jax --extra mlx pytest
```
