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
pnpm dev                                  # Vite dev server with sample data, http://localhost:1420
pnpm dev:real                             # the real core and judge in a browser: devbridge + Vite
pnpm dev:bridge                           # only the devbridge (whetstone-core over HTTP on 127.0.0.1:1421)
pnpm dev:http                             # only Vite, pointed at a running devbridge
pnpm tauri dev                            # desktop app with hot reload
pnpm build                                # type-check and build the front end
pnpm tauri build --debug                  # build and bundle the app (target/debug/whetstone)

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

The desktop app logs to stderr and to `~/Library/Logs/dev.whetstone.app/whetstone.log`:
the content root, catalog problems that were skipped and why, and every finished run.
Set `WHETSTONE_LOG=debug` to also log each command the window invokes. Builds made by
the Tauri CLI leave the sample data out of the bundle.

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
