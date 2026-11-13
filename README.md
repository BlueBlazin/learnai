# Whetstone

Whetstone is a native desktop app for mastering LLM inference and serving through
LeetCode-style practice problems. You write solutions in PyTorch, JAX, MLX, or plain
Python, and a local judge runs them against public and hidden tests.

Architecture and contracts: `docs/ARCHITECTURE.md`. Dependency versions: `docs/VERSIONS.md`.

## Prerequisites

- macOS 14 or later on Apple silicon (required by the MLX and PyTorch wheels)
- Node.js 25 and pnpm 10
- Rust (stable, 1.100 or later) with the Xcode Command Line Tools
- uv 0.9.26 or later (it uses Python 3.15, pinned in `python/.python-version`)

## Development

```sh
# Front end and desktop shell
pnpm install
pnpm dev                                  # Vite dev server only, http://localhost:1420
pnpm tauri dev                            # desktop app with hot reload
pnpm build                                # type-check and build the front end
pnpm tauri build --debug --no-bundle      # compile the app without bundling

# Rust workspace (src-tauri, crates/core, crates/devbridge)
cargo check --workspace
cargo test -p whetstone-core
cargo run -p whetstone-devbridge

# Python (always pass the full set of extras; a bare `uv sync` removes them)
uv sync --project python --extra jax --extra mlx
uv run --project python ws
uv run --project python pytest python/tests
uv run --project python ruff check python
```

## Layout

```
src/              React + TypeScript front end (shadcn/ui components in src/components/ui)
src-tauri/        Tauri shell (thin command layer)
crates/core/      whetstone-core: application logic
crates/devbridge/ dev-only HTTP bridge
python/           uv project: the whetstone package and the ws CLI
problems/         problem catalog
workspace/        your solutions (gitignored, created on demand)
```
