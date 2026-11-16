# Learn AI

Learn AI is a native desktop app for mastering LLM inference and serving through
LeetCode-style practice problems. You write solutions in PyTorch, JAX, MLX, or plain
Python, and a local judge runs them against public and hidden tests.

The catalog has 213 problems in 15 tracks, from stable softmax to capstones such as a
continuous-batching engine and a complete runtime for a real model, and a curated study
path (Inference 75). Catalog and path: `docs/CURRICULUM.md`. Writing problems:
`docs/PROBLEM_AUTHORING.md`.

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
pnpm dev                                  # Vite dev server with sample data, http://localhost:1420
pnpm dev:real                             # the real core and judge in a browser: devbridge + Vite
pnpm dev:bridge                           # only the devbridge (learnai-core over HTTP on 127.0.0.1:1421)
pnpm dev:http                             # only Vite, pointed at a running devbridge
pnpm tauri dev                            # desktop app with hot reload
pnpm build                                # type-check and build the front end
pnpm tauri build --debug                  # build and bundle the app (target/debug/learnai)

# Rust workspace (src-tauri, crates/core, crates/devbridge)
cargo check --workspace
cargo test -p learnai-core
cargo run -p learnai-devbridge

# Python (always pass the full set of extras; a bare `uv sync` removes them)
uv sync --project python --extra jax --extra mlx --extra models
uv run --project python --extra jax --extra mlx --extra models learnai
uv run --project python --extra jax --extra mlx --extra models learnai validate --all   # the catalog's quality gate
uv run --project python --extra jax --extra mlx --extra models pytest python/tests
uv run --project python ruff check python
```

The desktop app logs to stderr and to `~/Library/Logs/dev.learnai.app/learnai.log`:
the content root, catalog problems that were skipped and why, and every finished run.
Set `LEARNAI_LOG=debug` to also log each command the window invokes. Builds made by
the Tauri CLI leave the sample data out of the bundle.

## Layout

```
src/              React + TypeScript front end (shadcn/ui components in src/components/ui)
src-tauri/        Tauri shell (thin command layer)
crates/core/      learnai-core: application logic
crates/devbridge/ dev-only HTTP bridge
python/           uv project: the learnai package and the learnai CLI
problems/         problem catalog
workspace/        your solutions (gitignored, created on demand)
```
