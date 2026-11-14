# Whetstone dependency versions

**Verified on 2026-11-14.** Versions come from live registry queries (`npm view`, the crates.io API, the PyPI JSON API) run on that date; behavior notes come from the linked official docs. Anything not confirmed against a primary source is marked **UNVERIFIED**. Only stable releases are listed (pre-releases such as Tauri 3.0.0-alpha.6, notify 9.0.0-rc.8, and TypeScript 7.1 beta were ignored).

Local toolchain at verification time: uv 0.9.26 (latest is 0.13.8), Node 25.8.1, pnpm 10.33.0, rustc/cargo 1.100.0, Python 3.12/3.13/3.14.

## 1. Tauri

| Package | Version | Source |
|---|---|---|
| `tauri` (crate) | 2.12.5 | https://crates.io/crates/tauri |
| `tauri-build` (crate) | 2.7.4 | https://crates.io/crates/tauri-build |
| `tauri-cli` (crate, optional) | 2.12.4 | https://crates.io/crates/tauri-cli |
| `@tauri-apps/cli` | 2.12.4 | https://www.npmjs.com/package/@tauri-apps/cli |
| `@tauri-apps/api` | 2.12.4 | https://www.npmjs.com/package/@tauri-apps/api |
| `create-tauri-app` | 4.7.5 (npm) / 4.8.0 (crate) | https://www.npmjs.com/package/create-tauri-app |
| `tauri-plugin-dialog` + `@tauri-apps/plugin-dialog` | 2.8.4 | https://crates.io/crates/tauri-plugin-dialog |
| `tauri-plugin-opener` + `@tauri-apps/plugin-opener` | 2.7.3 | https://crates.io/crates/tauri-plugin-opener |
| `tauri-plugin-store` + `@tauri-apps/plugin-store` | 2.5.1 | https://crates.io/crates/tauri-plugin-store |
| `tauri-plugin-window-state` + `@tauri-apps/plugin-window-state` | 2.5.0 | https://crates.io/crates/tauri-plugin-window-state |
| `@tauri-apps/plugin-fs` (if needed) | 2.7.1 | https://www.npmjs.com/package/@tauri-apps/plugin-fs |
| `@tauri-apps/plugin-shell` (if needed) | 2.6.0 | https://www.npmjs.com/package/@tauri-apps/plugin-shell |

Tauri 3 exists only as alpha (`3.0.0-alpha.6`, npm dist-tag `next`). Stay on 2.x.

## 2. Front end

| Package | Version | Source |
|---|---|---|
| `react` | 19.3.1 | https://www.npmjs.com/package/react |
| `react-dom` | 19.3.1 | https://www.npmjs.com/package/react-dom |
| `@types/react` / `@types/react-dom` | 19.3.0 / 19.3.0 | https://www.npmjs.com/package/@types/react |
| `@types/node` | 26.6.6 | https://www.npmjs.com/package/@types/node |
| `typescript` | 7.0.2 (see gotcha below) | https://www.npmjs.com/package/typescript |
| `vite` | 8.3.8 | https://www.npmjs.com/package/vite |
| `@vitejs/plugin-react` | 6.3.0 | https://www.npmjs.com/package/@vitejs/plugin-react |
| `tailwindcss` | 4.3.5 | https://www.npmjs.com/package/tailwindcss |
| `@tailwindcss/vite` | 4.3.5 | https://www.npmjs.com/package/@tailwindcss/vite |
| `shadcn` (CLI, also a runtime dep for `shadcn/tailwind.css`) | 4.23.0 | https://www.npmjs.com/package/shadcn |
| `@base-ui/react` (shadcn default primitives) | 1.9.0 | https://www.npmjs.com/package/@base-ui/react |
| `radix-ui` (unified package, alternative base) | 1.8.1 | https://www.npmjs.com/package/radix-ui |
| `cn` (shadcn's class helper, replaces clsx + tailwind-merge) | 0.5.2 | https://www.npmjs.com/package/cn |
| `class-variance-authority` | 0.7.1 | https://www.npmjs.com/package/class-variance-authority |
| `lucide-react` | 1.72.0 | https://www.npmjs.com/package/lucide-react |
| `tw-animate-css` | 1.4.0 | https://www.npmjs.com/package/tw-animate-css |
| `clsx` (no longer required by new shadcn projects) | 2.1.1 | https://www.npmjs.com/package/clsx |
| `tailwind-merge` (no longer required by new shadcn projects) | 3.7.0 | https://www.npmjs.com/package/tailwind-merge |

Do not use `@base-ui-components/react` (stuck at 1.0.0-rc.0); the package was renamed to `@base-ui/react`.

## 3. App libraries

| Package | Version | Source |
|---|---|---|
| `@tanstack/react-router` | 1.170.46 | https://www.npmjs.com/package/@tanstack/react-router |
| `@tanstack/router-plugin` (Vite plugin, file-based routes) | 1.169.0 | https://www.npmjs.com/package/@tanstack/router-plugin |
| `@tanstack/react-query` | 5.107.1 | https://www.npmjs.com/package/@tanstack/react-query |
| `zustand` | 5.1.0 | https://www.npmjs.com/package/zustand |
| `react-resizable-panels` (what shadcn Resizable wraps) | 4.22.0 | https://www.npmjs.com/package/react-resizable-panels |
| `cmdk` | 1.1.1 | https://www.npmjs.com/package/cmdk |
| `sonner` | 2.0.8 | https://www.npmjs.com/package/sonner |

`next-themes` (0.4.6) is not needed: the shadcn Vite dark-mode guide uses a hand-written `ThemeProvider`.

## 4. Editor (CodeMirror 6)

| Package | Version | Source |
|---|---|---|
| `codemirror` | 6.0.2 | https://www.npmjs.com/package/codemirror |
| `@codemirror/state` | 6.7.6 | https://www.npmjs.com/package/@codemirror/state |
| `@codemirror/view` | 6.43.15 | https://www.npmjs.com/package/@codemirror/view |
| `@codemirror/language` | 6.13.1 | https://www.npmjs.com/package/@codemirror/language |
| `@codemirror/lang-python` | 6.2.1 | https://www.npmjs.com/package/@codemirror/lang-python |
| `@codemirror/commands` | 6.11.1 | https://www.npmjs.com/package/@codemirror/commands |
| `@codemirror/search` | 6.7.2 | https://www.npmjs.com/package/@codemirror/search |
| `@codemirror/autocomplete` | 6.20.4 | https://www.npmjs.com/package/@codemirror/autocomplete |
| `@codemirror/lint` | 6.9.8 | https://www.npmjs.com/package/@codemirror/lint |
| `@lezer/highlight` | 1.2.5 | https://www.npmjs.com/package/@lezer/highlight |
| `@uiw/react-codemirror` | 4.25.12 | https://www.npmjs.com/package/@uiw/react-codemirror |

`@uiw/react-codemirror` is current and maintained (last publish 2026-09-24; peers: React >=17, `codemirror` >=6). It also peer-depends on `@babel/runtime` and `@codemirror/theme-one-dark`, which pnpm will ask you to install. A thin custom hook around `EditorView` is a reasonable alternative if those extra peers are unwanted.

## 5. Markdown

| Package | Version | Source |
|---|---|---|
| `react-markdown` | 10.1.0 | https://www.npmjs.com/package/react-markdown |
| `remark-gfm` | 4.0.1 | https://www.npmjs.com/package/remark-gfm |
| `remark-math` | 6.0.0 | https://www.npmjs.com/package/remark-math |
| `rehype-katex` | 7.0.1 | https://www.npmjs.com/package/rehype-katex |
| `katex` | 0.19.1 | https://www.npmjs.com/package/katex |
| `shiki` | 4.6.0 | https://www.npmjs.com/package/shiki |
| `@shikijs/rehype` | 4.6.0 | https://www.npmjs.com/package/@shikijs/rehype |
| `react-shiki` (optional, third-party) | 0.11.3 | https://www.npmjs.com/package/react-shiki |

## 6. Fonts

| Package | Version | Source |
|---|---|---|
| `@fontsource-variable/inter` | 5.3.0 | https://www.npmjs.com/package/@fontsource-variable/inter |
| `@fontsource-variable/jetbrains-mono` | 5.3.0 | https://www.npmjs.com/package/@fontsource-variable/jetbrains-mono |
| `@fontsource-variable/geist` / `geist-mono` | 5.3.0 | https://www.npmjs.com/package/@fontsource-variable/geist |
| `@fontsource/inter` (static, for comparison) | 5.3.0 | https://www.npmjs.com/package/@fontsource/inter |

Fontsource is still the standard npm route to self-hosted open fonts; the files ship inside the package, so Vite bundles them and the app works offline. Docs: https://fontsource.org/docs/getting-started/variable

## 7. Rust crates

| Crate | Version | Source |
|---|---|---|
| `serde` | 1.0.229 | https://crates.io/crates/serde |
| `serde_json` | 1.0.154 | https://crates.io/crates/serde_json |
| `toml` | 1.2.1 | https://crates.io/crates/toml |
| `rusqlite` (use `features = ["bundled"]`) | 0.42.0 | https://crates.io/crates/rusqlite |
| `tokio` | 1.54.1 | https://crates.io/crates/tokio |
| `thiserror` | 2.0.21 | https://crates.io/crates/thiserror |
| `anyhow` | 1.0.104 | https://crates.io/crates/anyhow |
| `notify` | 8.2.2 (9.0.0-rc.8 is pre-release) | https://crates.io/crates/notify |
| `axum` | 0.8.9 | https://crates.io/crates/axum |
| `tower-http` | 0.7.1 | https://crates.io/crates/tower-http |
| `uuid` | 1.29.2 | https://crates.io/crates/uuid |
| `chrono` | 0.4.45 | https://crates.io/crates/chrono |
| `time` | 0.3.55 | https://crates.io/crates/time |
| `tracing` | 0.1.44 | https://crates.io/crates/tracing |
| `tracing-subscriber` | 0.3.23 | https://crates.io/crates/tracing-subscriber |

UNVERIFIED: that `axum` 0.8.9 and `tower-http` 0.7.1 are compatible with each other (tower-http 0.7 is newer than the last axum release, 2026-04). Check with `cargo tree -d` after adding both. UNVERIFIED: that the SQLite version bundled by rusqlite 0.42 is the one you want; the feature name `bundled` was not re-checked against the 0.42 docs.

## 8. Python

| Package | Version | Requires-Python | macOS arm64 wheels (CPython tags) | Source |
|---|---|---|---|---|
| `torch` | 2.15.0 | >=3.11 | cp311, cp312, cp313, cp314, cp315 (macOS 14.0+) | https://pypi.org/project/torch/ |
| `jax` | 0.12.0.1 | >=3.12 | pure Python | https://pypi.org/project/jax/ |
| `jaxlib` | 0.12.0.1 | >=3.12 | cp312, cp313, cp314, cp315 (macOS 11.0+) | https://pypi.org/project/jaxlib/ |
| `mlx` | 0.32.4 | >=3.10 | cp310 through cp315 (macOS 14.0+) | https://pypi.org/project/mlx/ |
| `huggingface_hub` (`models` extra) | 2.4.0 | | pure Python | https://pypi.org/project/huggingface-hub/ |
| `safetensors` (`models` extra) | 0.8.0 | | abi3 | https://pypi.org/project/safetensors/ |
| `tokenizers` (`models` extra) | 0.23.3 | | abi3 | https://pypi.org/project/tokenizers/ |
| `transformers` (`models` extra) | 5.23.0 | | pure Python | https://pypi.org/project/transformers/ |
| `mlx-lm` (`models` extra, macOS only) | 0.32.2 | >=3.11 | pure Python | https://pypi.org/project/mlx-lm/ |
| `numpy` | 2.5.4 | >=3.12 | cp312, cp313, cp314, cp315 | https://pypi.org/project/numpy/ |
| `pytest` | 9.2.0 | >=3.10 | pure Python | https://pypi.org/project/pytest/ |
| `httpx` | 0.28.1 | >=3.8 | pure Python | https://pypi.org/project/httpx/ |
| `uvicorn` | 0.54.0 | >=3.10 | pure Python | https://pypi.org/project/uvicorn/ |
| `starlette` | 1.10.0 | >=3.11 | pure Python | https://pypi.org/project/starlette/ |
| `safetensors` | 0.8.0 | >=3.10 | cp310-abi3 (works on all later versions) | https://pypi.org/project/safetensors/ |
| `tokenizers` | 0.23.3 | >=3.10 | cp310-abi3 | https://pypi.org/project/tokenizers/ |
| `huggingface_hub` | 2.4.0 | >=3.10 | pure Python | https://pypi.org/project/huggingface_hub/ |

## 9. uv

| Tool | Version | Source |
|---|---|---|
| `uv` | 0.13.8 (released 2026-11-12) | https://github.com/astral-sh/uv/releases/latest |

---

## Setup notes

### Tauri

Scaffold (non-interactive; documented in the create-tauri-app README, https://github.com/tauri-apps/create-tauri-app):

```sh
pnpm create tauri-app whetstone --template react-ts
cd whetstone
pnpm install
pnpm tauri dev
```

- The project name is a positional argument (`.` scaffolds into the current directory) and `--template react-ts` selects React + TypeScript + Vite. The README documents only these two for non-interactive use. `--manager pnpm`, `--identifier <id>`, `--yes`, and `--force` are confirmed by `--help` on 4.7.5 and were used to scaffold this repository. Running through `pnpm create` selects pnpm as the manager.
- Gotcha: the `react-ts` template currently pins `typescript ~6.0.3`, `vite ^8.0.16`, `@vitejs/plugin-react ^6.0.2`, `react ^19.1.0`, and includes `@tauri-apps/plugin-opener`. It lags the plain Vite template (which is on `typescript ~7.0.2`). Bump deliberately after scaffolding.
- Gotcha: `.` scaffolding into `/Users/sanjeetdasharath/learnai` will run into the existing `docs/` directory. The CLI has `-f, --force` for non-empty directories; this repository was scaffolded into a temp directory and moved in instead.
- Add plugins with `pnpm tauri add dialog` (and `store`, `window-state`, `opener`), which installs both the crate and the npm package and registers the plugin. UNVERIFIED on 2.12.4 but long-standing CLI behavior; the manual route is `cargo add tauri-plugin-dialog` plus `pnpm add @tauri-apps/plugin-dialog` plus a capability entry.

Streaming from Rust to the front end: use **Channels** (`tauri::ipc::Channel<T>` passed as a command argument, `new Channel<T>()` with `onmessage` from `@tauri-apps/api/core`). Doc: https://tauri.app/develop/calling-frontend/

- The docs state the event system "is not designed for low latency or high throughput situations", that event payloads are always JSON strings, and that async listeners may process rapid events out of order.
- Channels "are designed to be fast and deliver ordered data" and are what Tauri uses internally for download progress, child process output, and WebSocket messages.
- Use events (`emit` / `listen`) only for low-rate, app-wide notifications. Use one Channel per run for judge stdout, per-test results, and token streams.

### TypeScript 7 gotcha

`typescript@latest` is 7.0.2, the native (Go) compiler. `baseUrl` has been removed (error TS5102), and `moduleResolution: node10` is gone. Sources for this are the TypeScript issue tracker and third-party migration guides rather than the official release notes page (which returned 404 when fetched), so treat the details as partly **UNVERIFIED**.

Consequence: the shadcn Vite guide still shows `"baseUrl": "."` in `tsconfig.json` and `tsconfig.app.json`. On TS 7, omit `baseUrl` and keep only:

```json
"paths": { "@/*": ["./src/*"] }
```

Two workable choices: accept the Tauri template's `typescript ~6.0.3` for now (baseUrl is deprecated but accepted), or go to 7.0.2 and drop `baseUrl`. UNVERIFIED: whether every TypeScript-API-dependent tool in the stack (for example the TanStack Router plugin's generator and shadcn CLI's tsconfig reader) is fully happy on TS 7; the current create-vite `react-ts` template does ship `typescript ~7.0.2`, which is a good sign. Verified in this repository on 7.0.2: `tsc`, Vite 8.3.8, `shadcn init`, and `shadcn add` all work with `paths` and no `baseUrl`. The TanStack Router plugin is still unverified.

### shadcn/ui

Docs: https://ui.shadcn.com/docs/installation/vite, https://ui.shadcn.com/docs/cli, https://ui.shadcn.com/docs/theming, https://ui.shadcn.com/docs/dark-mode/vite

**Default primitive library: Base UI** (since the July 2026 "Base UI as the Default" changelog entry). `shadcn init` selects Base UI unless told otherwise, the docs open on the Base UI tab, and the docs recommend Base UI for new projects. Radix remains fully supported and receives the same components; pass `-b radix` to choose it. A third base, `aria` (React Aria), also exists.

Because the project is scaffolded by create-tauri-app, use the "Existing Project" path:

```sh
# 1. Tailwind
pnpm add tailwindcss @tailwindcss/vite
#    src/index.css (replace contents):  @import "tailwindcss";

# 2. Path alias: add "paths": { "@/*": ["./src/*"] } to tsconfig.json and
#    tsconfig.app.json (the docs also show "baseUrl": "." -- see TS 7 gotcha)

# 3. Vite config
pnpm add -D @types/node
#    vite.config.ts: plugins: [react(), tailwindcss()],
#    resolve.alias: { "@": path.resolve(__dirname, "./src") }

# 4. Init (interactive prompts cover base, preset, and more)
pnpm dlx shadcn@latest init
#    or pin the choices (without -p the CLI still prompts for a preset, even with -y):
pnpm dlx shadcn@latest init -b base -p nova -y

# 5. Components
pnpm dlx shadcn@latest add button resizable command sonner
```

`init` flags (from the CLI docs): `-t, --template <next|vite|start|react-router|laravel|astro>`, `-b, --base <base|radix|aria>`, `-p, --preset [name]`, `-y, --yes` (default true), `-d, --defaults` (equals `--template=next --preset=nova`, so do **not** use it here), `-f, --force`, `-c, --cwd`, `-n, --name`, `-s, --silent`, `--css-variables` / `--no-css-variables` (default on), `--monorepo` / `--no-monorepo`, `--rtl` / `--no-rtl`, `--pointer` / `--no-pointer`, `--reinstall` / `--no-reinstall`. `shadcn create` is an alias of `init`. Presets can be designed at https://ui.shadcn.com/create and passed as `--preset <CODE>`.

Gotchas:

- `pnpm dlx shadcn@latest init -t vite` scaffolds a brand-new Vite project. Do not use `-t` inside the Tauri project; it is for greenfield web apps.
- The Tauri template has no `tsconfig.app.json` split in older template versions. Confirmed for 4.7.5: the template has only `tsconfig.json` and `tsconfig.node.json`; the alias lives in `tsconfig.json`.
- Dependencies shadcn currently requires (manual-install page): `shadcn class-variance-authority cn lucide-react tw-animate-css`. Since September 2026 the `cn` npm package replaces the `clsx` + `tailwind-merge` pair; `lib/utils.ts` is now `export { cn } from "cn"`. Existing projects migrate with `pnpm dlx shadcn@latest migrate cn`.
- Resizable wraps `react-resizable-panels` v4. The shadcn wrappers are unchanged, but the v4 primitives were renamed: `PanelGroup` to `Group`, `PanelResizeHandle` to `Separator`, `direction` to `orientation`, `defaultSize={50}` to `defaultSize="50%"`, `onLayout` to `onLayoutChange`.

Theming in the current version:

- Global CSS begins with `@import "tailwindcss"; @import "tw-animate-css"; @import "shadcn/tailwind.css";` followed by `@custom-variant dark (&:is(.dark *));`.
- Tokens are CSS variables in `oklch()`, defined under `:root` (light) and overridden under `.dark`: `--background`/`--foreground`, and `--card`, `--popover`, `--primary`, `--secondary`, `--muted`, `--accent`, `--destructive`, each with a `-foreground` partner, plus `--border`, `--input`, `--ring`, and `--radius` (default `0.625rem`, with `--radius-*` derived by `calc()`).
- An `@theme inline { --color-primary: var(--primary); ... }` block maps the variables to Tailwind utilities (`bg-primary`, and so on). There is no `tailwind.config.js` in Tailwind 4; `components.json` leaves `tailwind.config` empty.
- Base colors: Neutral, Stone, Zinc, Mauve, Olive, Mist, Taupe. `tailwind.cssVariables` defaults to `true` and is an install-time choice.
- Dark mode on Vite: a hand-written `ThemeProvider` (code in the dark-mode doc) toggles a `light`/`dark` class on `<html>`, supports `system`, and persists to `localStorage` under `vite-ui-theme`. No `next-themes`.

### Markdown, math, and highlighting

- `react-markdown` 10 `<Markdown>` is synchronous. `remarkPlugins={[remarkGfm, remarkMath]}` and `rehypePlugins={[rehypeKatex]}` work directly; import `katex/dist/katex.min.css` yourself (rehype-katex does not).
- Shiki: the default `@shikijs/rehype` plugin is async, so it does not work in the plain `<Markdown>` component. Two documented options:
  1. Recommended: create one highlighter at app start with `createHighlighterCore` (only the `python`, `bash`, `json`, `rust` grammars and two themes, with an explicit engine), then use `rehypeShikiFromHighlighter` from `@shikijs/rehype/core`, which can run synchronously inside `<Markdown>`. Pass `themes: { light, dark }` for dual themes. Docs: https://shiki.style/packages/rehype
  2. Use `MarkdownHooks` from `react-markdown` (client-side async via `useEffect`, with a `fallback` prop) with the default async plugin.
- `react-shiki` 0.11.3 (a `components.code` override) is a third-party convenience and not needed.

### Fonts

```sh
pnpm add @fontsource-variable/inter @fontsource-variable/jetbrains-mono
```

```ts
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
```

- Variable fonts live under the `@fontsource-variable/<font-id>` scope; static fonts under `@fontsource/<font-id>`.
- The CSS family name gets a "Variable" suffix: `font-family: "Inter Variable", sans-serif`. (The docs show this pattern for Nunito Sans; the suffix for other fonts is inferred from that convention.)
- Use one import per font. Extra axes are chosen by file (`wght-italic.css`, `opsz.css`, `standard.css`, `full.css`), not by stacking imports.

### Python version and frameworks

**Newest Python minor with macOS arm64 wheels for torch, jax/jaxlib, and mlx: 3.15** (all three publish `cp315` wheels; CPython 3.15.0 shipped 2026-10-09, latest 3.15.2). numpy 2.5.4 also has cp315 wheels.

**Pinned: Python 3.15** (`requires-python = ">=3.15,<3.16"`, `.python-version` = `3.15`).

- Verified on this machine (Apple M1 Max, CPython 3.15.2, uv 0.9.26): `uv sync --extra jax --extra mlx` resolves and installs, and `import torch, jax, mlx.core` works with torch 2.15.0 (MPS available), jax 0.12.0.1, and mlx 0.32.4.
- uv 0.9.26 does not offer a stable managed 3.15 download (only `3.15.0a5`), so it uses a system interpreter (here Homebrew's `python@3.15`). A newer uv can fetch a managed 3.15.
- Fallback if a later dependency lacks cp315 wheels: 3.14 has wheels for every package in the table. The hard floor is 3.12, set by jax, jaxlib, and numpy (`>=3.12`).

JAX on Apple Silicon: **treat it as CPU-only.** The JAX install page lists "Apple GPU: experimental" in its table but also states that JAX is not supported on Mac GPU and to use the CPU install (`pip install -U jax`). Apple's `jax-metal` plugin is dead: last release 0.1.1 on 2024-10-08, Apple's page lists it as currently not supported, and it is reported incompatible with jax >= 0.5.1. Third-party Metal/MLX-backed plugins exist (`jax-mps`, `metaljax`, `jax-metallib`) but are alpha/beta and lack float64; they are UNVERIFIED and not recommended for a judge. Sources: https://docs.jax.dev/en/latest/installation.html, https://developer.apple.com/metal/jax/, https://pypi.org/project/jax-metal/

MLX: requires Apple silicon, native arm64 Python >= 3.10, and **macOS >= 14.0** (https://ml-explore.github.io/mlx/build/html/install.html). Wheels are built per macOS major (14, 15, 26). PyTorch 2.15.0 macOS wheels also target macOS 14.0+.

Judge implications: PyTorch (MPS available) and MLX use the GPU; JAX runs on CPU. Timing-based problems must not compare JAX wall-clock against the others, and float64 oracles should run on CPU.

### uv

Installed uv is 0.9.26; latest is 0.13.8. Update before starting (`uv self update`, or `brew upgrade uv` if installed through Homebrew). UNVERIFIED: any behavior changes between 0.9 and 0.13 beyond what the current docs state; the experiment below ran on 0.9.26 and matches the current docs.

Sync semantics (doc: https://docs.astral.sh/uv/concepts/projects/sync/ and https://docs.astral.sh/uv/reference/cli/):

- `uv sync` is **exact** by default: it removes anything not selected for installation, and extras are not selected unless passed. So `uv sync --extra jax` after `uv sync --extra mlx` **uninstalls MLX**. `uv sync --inexact` keeps extraneous packages.
- `uv run` is **inexact** by default: it installs what is missing and removes nothing. So `uv run --extra jax ...` does **not** remove packages from other extras installed earlier. `uv run --exact` opts into removal.

Confirmed empirically in a scratch project (uv 0.9.26) with extras `a` and `b`:

| Step | Result |
|---|---|
| `uv sync --extra a` | `a` installed |
| `uv run --extra b ...` | `a` and `b` both present |
| `uv run ...` (no extras) | `a` and `b` both still present |
| `uv sync --extra b` | `a` removed |
| `uv sync --inexact --extra a` | `a` and `b` both present |
| `uv sync` | both removed |

Recommended way to keep one venv that accumulates optional frameworks:

```toml
[project]
requires-python = ">=3.15,<3.16"
dependencies = ["torch", "numpy", "pytest", "safetensors", "tokenizers", "huggingface_hub"]

[project.optional-dependencies]
jax = ["jax"]
mlx = ["mlx"]
serve = ["starlette", "uvicorn", "httpx"]
```

- The lockfile always resolves all extras together, so the set is guaranteed mutually compatible whether or not an extra is installed.
- Have the app record which extras the user enabled (for example in its settings store) and always pass the **full enabled set**: `uv sync --extra jax --extra mlx` to install, and `uv run --extra jax --extra mlx python -m judge ...` to execute. Passing the full set makes the result independent of exact/inexact defaults.
- Never call a bare `uv sync` (or `uv sync --extra X` with a partial set) from the app; it silently strips the other frameworks. If a partial call is unavoidable, add `--inexact`.
- For fast, side-effect-free judge runs once the environment is set up, `uv run --no-sync ...` skips the environment check entirely (it implies `--frozen`).
- Dependency groups (`[dependency-groups]`, `--group`) are the alternative mechanism; the `dev` group syncs by default. Extras fit better here because the frameworks are user-facing options.

### Linux hosts (milestone 4, remote runs)

No dependency was added or bumped for milestone 4. What changed is two markers in
`python/pyproject.toml`, and what follows was checked for the existing pins.

- `mlx` (extra `mlx`) and `mlx-lm` (extra `models`) carry
  `sys_platform == 'darwin' and platform_machine == 'arm64'`. `mlx` 0.32.4 does
  publish manylinux wheels (glibc 2.35), but the catalog's MLX problems need Metal
  (`requires = "mlx-gpu"`), so a Linux host installs no MLX and the same
  `--extra jax --extra mlx` can be passed everywhere. After the change `uv lock`
  moved no version: the only differences in `uv.lock` are the markers and the MLX
  wheels for Linux and Windows, which are no longer listed.
- `uv.lock` is still one universal lock (`uv lock --check` passes). For Linux it
  pins `torch` 2.15.0 from PyPI (`manylinux_2_28` wheels for x86_64 and aarch64,
  cp315), which is the CUDA build: it depends on `cuda-toolkit` 13.2.2, the
  `nvidia-*` runtime wheels (`nvidia-cudnn-cu13` 9.26.0.51 and others), and
  **`triton` 3.9.0**, all with `sys_platform == 'linux'` markers. So real Triton is
  present on a Linux host without being a dependency of ours
  (https://pypi.org/project/triton/).
- Verified on this Mac with uv 0.9.26, since no Linux machine was available:
  `uv export --frozen` of the lock, then `uv pip install --dry-run --require-hashes
  --python-version 3.15 --python-platform <p>` for `x86_64-manylinux_2_28` and
  `aarch64-manylinux_2_28`. With the extras a harness run uses (`jax`, `mlx`) and
  `--only-binary :all:`, both platforms resolve to 48 packages, every one a wheel.
  With `models` added (69 packages) one package has no Linux wheel for Python 3.15:
  **`pyyaml` 6.0.3** (needed by `huggingface_hub` and `transformers`) must be built
  from its sdist, so `ws assets fetch` on a Linux host needs a working build
  environment or a later `pyyaml`. `uv pip compile pyproject.toml --python-platform
  x86_64-manylinux_2_28` resolves to the same versions as the lock.
- NOT verified: an actual install or run on Linux (wheel contents, the CUDA runtime
  loading, glibc compatibility of the host, a managed Python 3.15 from uv on Linux),
  and anything on a GPU.
- `ws doctor --json` was run in a fresh environment with no extras
  (`UV_PROJECT_ENVIRONMENT=<tmp> uv run --frozen --no-dev ws doctor --json`): it
  reports `jax` and `mlx` as null and exits 0, and a torch problem is judged there.
- JAX's persistent compilation cache (`jax_compilation_cache_dir`,
  `jax_persistent_cache_min_compile_time_secs`,
  `jax_persistent_cache_min_entry_size_bytes`; JAX 0.12.0.1) is switched on by the
  harness. Its optional size bound (`jax_compilation_cache_max_size`, which needs
  `filelock`) is not used: with it every write rescans the directory, which doubled
  first-run time here (9.9 s to 18.5 s on `quantize-real-weights`).
