# Whetstone design system

Whetstone is a study tool someone will spend hundreds of hours in. The design
direction is **technical editorial**: the problem statement is set like a page
from a well-made textbook, the editor sits beside it, and everything else gets
out of the way.

This document is the reference for anyone adding UI. The tokens live in
`src/styles/tokens.css`; the reading surface in `src/styles/prose.css`; the
binding component rules in `docs/ARCHITECTURE.md` section 5.

## 1. Principles

1. **The statement is the product.** The statement, hints, and editorial are
   typeset prose on a fixed measure. Chrome never competes with them: one
   44px app bar, one 40px toolbar per pane, nothing else.
2. **Rules and whitespace, not boxes.** Structure comes from hairlines, a
   heavier ink rule above each section, and vertical rhythm. There are no cards,
   no drop shadows on resting surfaces, and no badges. The only filled blocks
   are code listings and program output.
3. **Words over ornaments.** Judges, frameworks, verdicts, and counts are
   written out ("Numeric, Property", "5 of 7 cases passed"). A mark appears only
   where it encodes something a word cannot at a glance: status, difficulty,
   verdict.
4. **Color means outcome, and never alone.** Hue is reserved for pass, fail,
   and warn, and each use also carries a distinct shape and a word.
5. **Honest and quiet.** Progress is a sentence and plain counts. No rings,
   confetti, or percentages in display type.

What this deliberately is not: an untouched shadcn neutral theme, Inter or
Geist everywhere, gradients, glass or blur, emoji as icons, card grids, pill
badges, or one large radius on everything.

## 2. Typography

Three families, each with one job. All are self-hosted through Fontsource and
bundled by Vite; the app makes no network requests for type.

| Role | Family | Why |
|---|---|---|
| Reading and titles | **Literata** (variable, optical size axis, roman and italic) | Designed for long-form reading on screens. Sturdy at 17px, and its optical sizes give titles finer detail without a second display face. |
| Interface | **IBM Plex Sans** (variable) | An engineered grotesque that is legible at 12 to 13px, has true tabular figures, and recedes next to a serif. |
| Code and output | **IBM Plex Mono** (400, 400 italic, 500, 600) | Same design family as the interface face, so chrome and code share proportions; its italic makes comments distinct without color. |

Math is set by KaTeX in its own faces, scaled (`1.04em`) to sit with Literata's
x-height.

### Scale

The interface sizes sit below the reading size; display sizes rise from it in
steps of roughly 1.2.

| Token | Size / line height | Use |
|---|---|---|
| `text-2xs` | 11 / 16 | rarely; dense annotations |
| `text-xs` | 12 / 16 | metadata, table secondary columns, key caps |
| `text-sm` | 13 / 20 | **default interface text**, controls |
| `text-base` | 14 / 22 | dialog titles |
| `text-md` | 16 / 24 | serif titles in lists, compact prose (hints) |
| `text-prose` | 17 / 28 | **statement body** |
| `text-lg` | 19 / 26 | prose `h2`, track headings, verdicts |
| `text-xl` | 23 / 30 | prose `h1` |
| `text-2xl` | 28 / 34 | page titles, problem title in a narrow pane |
| `text-3xl` | 36 / 40 | problem title, the home "continue" title |
| `text-code` | 13 / 21 | editor and listings |

### Prose rules (`.prose`)

- Measure: `--container-measure` (40rem, about 68 characters of body serif).
  Text is left-aligned and ragged right with `text-wrap: pretty`; headings use
  `text-wrap: balance`.
- Hierarchy within one family: `h2` is semibold at 19px; `h3` (used for
  "Example 1") is italic at body size, the way a textbook sets an example head.
- Inline code is the mono face with no pill behind it. Code blocks are the one
  tinted block (`--code`).
- Tables are booktabs: a 1.5px ink rule above and below, a hairline under the
  head, no row or column rules, tabular figures.
- Callouts (`> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`) are an indent with a
  colored hairline and a small sans label. Ordinary quotations use the same
  indent in italic. Nothing is boxed.
- Display math gets 1.25em of air above and below and scrolls sideways rather
  than shrinking. Punctuation after inline math is kept on the same line as the
  formula.
- Unordered lists use a short dash, ordered lists use tabular sans numerals in
  muted ink.

## 3. Color

The palette comes from the subject: a whetstone and tempered steel. Paper is a
cool stone white (not cream), ink is blue-black, and the single accent is the
blue that steel turns when it is tempered. A straw yellow, the first temper
color, is used only for highlighting (bracket and search matches).

All values are `oklch()`. Both themes are first-class: dark is blued steel, not
black, with the same hues lifted.

| Token | Light | Dark | Use |
|---|---|---|---|
| `background` | `0.972 0.004 255` | `0.185 0.012 260` | chrome: app bar, toolbars, sidebars |
| `paper` | `0.993 0.002 255` | `0.212 0.013 260` | reading and writing surfaces |
| `foreground` | `0.25 0.022 262` | `0.90 0.008 255` | ink |
| `muted-foreground` | `0.49 0.018 258` | `0.71 0.014 258` | secondary text (at least 4.5:1 on both surfaces) |
| `faint` | `0.66 0.012 258` | `0.52 0.014 258` | decoration only: line numbers, inactive marks |
| `border` | `0.895 0.007 255` | `0.295 0.014 260` | hairlines |
| `rule` | ink | `0.78 0.01 255` | section rules, booktabs rules |
| `primary` | `0.43 0.125 262` | `0.77 0.095 262` | temper blue: the Submit button, links, caret |
| `accent` | `0.94 0.008 258` | `0.275 0.016 260` | hover and selected rows |
| `code` | `0.96 0.005 255` | `0.24 0.013 260` | listings and output blocks |
| `pass` / `fail` / `warn` | `0.49 0.105 158` / `0.51 0.17 27` / `0.53 0.11 68` | `0.77 0.12 158` / `0.74 0.14 25` / `0.80 0.11 80` | outcomes only |
| `straw` | `0.93 0.075 96` | `0.38 0.055 92` | highlight |

Syntax uses four inks shared by the editor and static listings
(`--syntax-*`): keywords in temper blue, strings in green, numbers and
constants in violet, comments in italic gray. Definitions are bold body ink.
CodeMirror reads the variables through its theme and shiki through a
CSS-variables theme, so one render is correct in both modes.

## 4. Difficulty, status, and verdicts

These are the places where a generic app reaches for colored pills. Whetstone
does not.

- **Difficulty** is a magnitude, not an outcome, so it has no hue: a three-bar
  mark filled to the level, in ink, plus the word (`DifficultyMark`).
- **Status** is a shape: an empty ring (not started), a half-filled ring
  (attempted), a filled disc with a check (solved). Hue reinforces but the
  shapes alone are sufficient (`StatusMark`).
- **Verdicts and case results** each have their own icon and always their
  label (`VerdictMark`, `CaseMark`). Pass is green, wrong answers and errors
  are red, time limits and judge failures are amber, canceled is gray. Red and
  green are never the only difference between two states.
- **Metrics over a limit** show the value in red, a warning icon, and the words
  "over the limit of" (`MetricList`).

## 5. Space, radii, elevation

- Spacing is Tailwind's 4px grid. Named values: `titlebar` 44px, `toolbar`
  40px, `gutter` 40px (page margins; 24px in a narrow pane).
- Radii are small and few: `sm` 2px (focus outlines on text), `md` 4px
  (controls, listings), `lg` 5px, `xl` 7px (popovers and dialogs). Nothing is
  fully rounded except status discs.
- One elevation (`--shadow-float`), used only by layers that float: menus,
  popovers, dialogs, the command palette, editor tooltips.
- Density: table rows are about 36px, list rows 32px, controls 28px (`sm`) in
  toolbars.

## 6. Motion

- Transitions are 120ms with `cubic-bezier(0.2, 0, 0, 1)`, and only answer an
  action: hover, press, a disclosure opening, a popover appearing.
- Nothing animates on its own except the spinner on a running case and the run
  progress rule. There are no entrance animations and no staggered reveals.
- `prefers-reduced-motion` removes all transitions and animations.

## 7. Native window behavior

- The document never scrolls or rubber-bands; panes scroll inside
  `ScrollArea`.
- Chrome is not text-selectable and uses the default cursor. Reading surfaces,
  code, output, and paths opt in with `.selectable`.
- Focus is one 2px ring in `--ring`, keyboard only.
- The web context menu is suppressed on chrome inside Tauri and kept where text
  can be selected or edited. Web links open in the system browser.
- The app bar carries `data-tauri-drag-region`. With an overlay title bar
  (`VITE_TITLEBAR_OVERLAY=1`, mirroring `tauri.conf.json`) it reserves 96px on
  the left for the traffic lights.
- Shortcuts: Mod-K command palette, Mod-Enter run, Shift-Mod-Enter submit, `/`
  focuses library search.

## 8. Components

Primitives are shadcn/ui on Base UI in `src/components/ui/`, restyled through
the tokens. Local changes to generated primitives: `tabs` is rewritten as
text tabs with an underline (no pill variant); `progress` is a 2px rule;
`table` rows are hairline-separated with tighter cells; dialogs have no
backdrop blur; `command` items have no trailing check; tooltips wait 350ms.

Composed components, by area (all shown in their states at
`/#/dev/components` in development):

| Area | Components |
|---|---|
| `app/` | `AppShell`, `CommandPalette`, `AppLink`, `IconButton`, `ActionButton`, `ConfirmDialog`, `ShortcutHint`, `Page`, `PageHeader`, `Section`, `MetaList` / `MetaItem`, `EmptyState`, `ErrorState`, `LoadingLines` |
| `prose/` | `Prose`, `CodeBlock` |
| `editor/` | `CodeEditor` |
| `problem/` | `DifficultyMark`, `StatusMark`, `JudgeList`, `FrameworkList`, `FrameworkSwitcher`, `ProgressRule`, `ProblemTable`, `ProblemHeader`, `HintList`, `EditorialPanel`, `SubmissionList`, `SubmissionView` |
| `results/` | `ResultsPanel`, `VerdictHeader`, `VerdictMark`, `CaseMark`, `CaseRow` / `CaseList`, `MetricList`, `OutputBlock`, `RunProgress` |
| `library/` | `LibraryFilterBar`, `FilterSelect`, `TrackList`, `TrackSection` |
| `workbench/` | `StatementPane`, `EditorPane`, `SaveIndicator` |
| `home/` | `ContinueEntry`, `ProgressSummary`, `TrackProgress` |
| `settings/` | `EnvironmentPanel`, `ThemePicker`, `PathField` |

Conventions:

- A pattern used twice is a component. Look in the table above before writing
  markup in a route.
- Routes (`src/routes/`) contain layout and data wiring only.
- In-app links are built from `src/lib/routes.ts` and rendered with `AppLink`.
- User-facing strings for enums live in `src/lib/labels.ts`; number, time, and
  metric formatting in `src/lib/format.ts`.
- Do not name a directory `workspace`: the repository's `.gitignore` ignores
  that name, and Tailwind skips ignored paths when scanning for classes.

### Known exceptions

- The search panel inside CodeMirror (Mod-F) is CodeMirror's own DOM, themed
  from the tokens. It is the one interactive surface not built from a shadcn
  primitive.
- Generated primitives in `src/components/ui/` keep shadcn's own arbitrary
  values where they exist; app components use tokens only.

## 9. Authoring conventions the UI relies on

- Statements, hints, and editorials are Markdown with GFM tables, `$...$` and
  `$$...$$` math, and fenced code with a language tag (`python`, `json`,
  `bash`). Untagged fences render as plain listings.
- Callouts use GitHub alert syntax: `> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`.
- Use `##` for sections and `### Example N` for worked examples; the title is
  rendered by the app, so a statement should not start with an `#` heading.
