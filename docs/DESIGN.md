# Learn AI design system

Learn AI is a study tool someone will spend hundreds of hours in. The design
direction is **technical editorial**: the problem statement is set like a page
from a well-made textbook, the editor sits beside it, and everything else gets
out of the way. Around that quiet desk the app has a character of its own: the
curriculum is organized as areas (subjects) made of modules, each module an
illustrated place with its color, and Home shows where on the route through
them the learner stands.

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
   are code listings, program output, and the scenes.
3. **Words over ornaments.** Judges, frameworks, verdicts, and counts are
   written out ("Numeric, Property", "5 of 7 cases passed"). A mark appears only
   where it encodes something a word cannot at a glance: status, difficulty,
   verdict.
4. **Hue has two jobs, kept apart.** Outcome colors (pass, fail, warn) say how
   something went, and each use also carries a distinct shape and a word. Module
   colors say which module something belongs to, and are never used for an
   outcome. Nothing else is colored.
5. **Calm at the desk, character around it.** The workspace (statement, editor,
   results) stays as quiet as a page. Home and the Curriculum screens are where
   the app shows warmth: the scenes, the module color bar, the streak and the
   weekly goal. Progress is still honest: a sentence and plain counts, a pip per
   core problem. No rings, confetti, percentages in display type, levels, or locks.
6. **Pictures set a mood; they do not explain.** A scene is a place on the
   route, not a diagram of the module's topic. Icons that tried to depict a KV
   cache or a GPU were tried and dropped: illustration of abstract technical
   ideas reads as clip art. Teaching figures belong in statements, drawn by
   authors (`Figure`).

What this deliberately is not: an untouched shadcn neutral theme, Inter or
Geist everywhere, gradients, glass or blur, emoji as icons, card grids, pill
badges, or one large radius on everything.

## 2. Typography

Three families, each with one job, and a fourth for the wordmark. All are self-hosted through Fontsource and
bundled by Vite; the app makes no network requests for type.

| Role | Family | Why |
|---|---|---|
| Reading and titles | **Literata** (variable, optical size axis, roman and italic) | Designed for long-form reading on screens. Sturdy at 17px, and its optical sizes give titles finer detail without a second display face. |
| Interface | **IBM Plex Sans** (variable) | An engineered grotesque that is legible at 12 to 13px, has true tabular figures, and recedes next to a serif. |
| Code and output | **IBM Plex Mono** (400, 400 italic, 500, 600) | Same design family as the interface face, so chrome and code share proportions; its italic makes comments distinct without color. |

The wordmark in the title bar is the one exception: "Learn AI" is set in
**Google Sans** bold (700, Latin only), which is used nowhere else.

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
  head, no row or column rules, tabular figures. Code in a cell never breaks
  mid-token; a table wider than the measure scrolls sideways.
- Listings wider than the measure scroll sideways inside their block. Nothing in
  prose is ever clipped.
- Callouts (`> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`) are an indent with a
  colored hairline and a small sans label. Ordinary quotations use the same
  indent in italic. Nothing is boxed.
- Display math gets 1.25em of air above and below and scrolls sideways rather
  than shrinking. Punctuation after inline math is kept on the same line as the
  formula.
- Unordered lists use a short dash, ordered lists use tabular sans numerals in
  muted ink.
- Figures are numbered and captioned the way a textbook sets them: the drawing
  flush left on the measure, then "Figure N." in semibold ink and the caption
  in the interface sans at 13px, in secondary ink. An image alone on its
  paragraph is a figure; its alt text is the caption and may contain math. An
  author's SVG is sanitized (shapes and text only) and inlined, so its
  `currentColor` ink and the figure palette follow the theme. It is scaled to
  the measure, to at most 4/3 of its natural size: a drawing 480 units wide
  fills the 640px measure, and text drawn at 13 units stays between about 12
  and 17 pixels across pane widths.

## 3. Color

The palette comes from sharpening tools: stone and tempered steel. Paper is a
cool stone white (not cream), ink is blue-black, and the single accent is the
blue that steel turns when it is tempered. A straw yellow, the first temper
color, is used only for highlighting (bracket and search matches).

All values are `oklch()`. Both themes are first-class. Dark is charcoal: its
surfaces and text are neutral grays with no hue of their own (chroma 0),
near-black chrome under charcoal paper, so the only hue on screen is the accent
and the outcome colors, lifted from the light theme. Body text sits near 15:1
on paper and secondary text near 8:1.

| Token | Light | Dark | Use |
|---|---|---|---|
| `background` | `0.972 0.004 255` | `0.165 0 0` | chrome: app bar, toolbars, sidebars |
| `paper` | `0.993 0.002 255` | `0.20 0 0` | reading and writing surfaces |
| `foreground` | `0.25 0.022 262` | `0.93 0 0` | ink |
| `muted-foreground` | `0.49 0.018 258` | `0.74 0 0` | secondary text (at least 4.5:1 on both surfaces) |
| `faint` | `0.66 0.012 258` | `0.55 0 0` | decoration only: line numbers, inactive marks |
| `border` | `0.895 0.007 255` | `0.29 0 0` | hairlines |
| `rule` | ink | `0.80 0 0` | section rules, booktabs rules |
| `primary` | `0.43 0.125 262` | `0.78 0.10 262` | temper blue: the Submit button, links, caret |
| `accent` | `0.94 0.008 258` | `0.27 0 0` | hover and selected rows |
| `code` | `0.96 0.005 255` | `0.235 0 0` | listings and output blocks (utility `bg-listing`; see below) |
| `pass` / `fail` / `warn` | `0.49 0.105 158` / `0.51 0.17 27` / `0.53 0.11 68` | `0.77 0.12 158` / `0.74 0.14 25` / `0.80 0.11 80` | outcomes only |
| `straw` | `0.93 0.075 96` | `0.38 0.055 92` | highlight |

Figures drawn by problem authors get a palette of their own (`--figure-*`), so
a drawing never invents a color: `ink`, `muted`, and `faint` are the text,
secondary, and decoration inks, `accent` is temper blue, and `straw` is the
other temper color at a strength that reads as a line (`0.56 0.115 72` light,
`0.82 0.105 84` dark). Blue and straw differ in lightness as well as hue, and
figures label their series, so the pair survives color-vision deficiency.
`--figure-sans`, `--figure-serif`, and `--figure-mono` are the three families.

The listing surface is exposed to Tailwind as `listing`, not `code`: `text-code`
is the code type size, and a color token of the same name would make that one
utility set the text to the color of its own block.

### Module colors

Each module (a track) may carry a color (`color` in `track.toml`, see
`docs/ARCHITECTURE.md` section 3.3). It is the dominant ink of that module's
scene, so the picture and the color are one identity. `src/lib/module-colors.ts`
turns an area's modules into custom properties on the screen's root
(`moduleColorVars`): `--module-N`, for module number N of the area, is the
color at full saturation, for the fills of the module progress bar and the flat
plate of a module with no scene, on both themes; `--module-N-mark` is the same
color for pips, darkened on the light theme until it holds 3:1 against paper. A
module name is never set in its color (yellow and lime cannot reach text
contrast); a square `ModuleSwatch` sits beside ordinary ink instead. The swatch
takes the module's own color and applies the same darkening, so it works
anywhere, including the Library's track list and the workspace header, which
have no area around them. A module with no color falls back to the accent.

### Scenes and pixel icons

- **Scenes** (`src/assets/scenes/`, looked up by slug through
  `src/lib/scenes.ts`) are flat screenprint landscapes in the manner of
  mid-century travel posters: five flat inks, no gradients, no texture, no
  outlines. Every scene shares off-white paper, blue-black ink, and steel blue,
  and adds one saturated ink of its own. There are fifteen, one per module of
  "Inference and serving" (`scene` in `track.toml`), running from a trailhead
  at dawn to a summit observatory at night. Fourteen are empty landscapes; one,
  the CUDA module's, deliberately shows a person, as an homage the owner asked
  for. It is the exception, not a direction for the series. They were generated
  from one style reference; `tools/scene-art/` holds the prompts and scripts,
  and its README says how to add or redo one so that it stays in the series.
- A scene is shown whole and unfiltered: no overlay text, no dimming for modules
  not yet reached, no grayscale. Color is there from the first day; progress is
  shown by pips and marks beside the picture.
- An area has no scene of its own: its banner (`AreaBanner`) is one equal
  vertical slice of each module's scene, side by side in module order.
- **Pixel icons** (`PixelIcon`, `src/assets/pixel/`): a flame for the streak
  and a star for a goal met or a module with every problem solved, 24x24
  sprites drawn at whole multiples with
  `image-rendering: pixelated`. They are generic game symbols, not concept
  art, and there are only these two.

Syntax uses four inks shared by the editor and static listings
(`--syntax-*`): keywords in temper blue, strings in green, numbers and
constants in violet, comments in italic gray. Definitions are bold body ink.
CodeMirror reads the variables through its theme and shiki through a
CSS-variables theme, so one render is correct in both modes.

## 4. Difficulty, status, and verdicts

These are the places where a generic app reaches for colored pills. Learn AI
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
- **Prerequisites** are advice, never a gate. The problem header lists what a
  problem builds on, each with its status mark and a link (`PrerequisiteList`).
  In lists, a problem whose prerequisites are not solved yet carries the words
  "best after ..." beside its title in secondary ink (`PrerequisiteNote`): no
  lock, no dimming, and the row opens like any other.
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
- A theme change is instant. `stores/theme.ts` turns transitions off (the
  `theme-switching` class) for the frame in which the colors change; otherwise
  every control's hover transition would run at once, and restart at each level
  of nesting.
- Nothing animates on its own except the spinner on a running case and the run
  progress rule. There are no entrance animations and no staggered reveals.
- `prefers-reduced-motion` removes all transitions and animations.

## 7. Native window behavior

- The document never scrolls or rubber-bands (`overflow: clip` on the root, so
  not even focus can scroll it); panes scroll inside `ScrollArea`.
- Chrome is not text-selectable and uses the default cursor. Reading surfaces,
  code, output, and paths opt in with `.selectable`.
- Focus is one ring in `--ring`, keyboard only: the primitives' 3px halo on
  controls, a 2px outline on everything else that can take focus.
- The web context menu is suppressed on chrome inside Tauri and kept where text
  can be selected or edited. Web links open in the system browser.
- Title bar. `tauri.conf.json` sets `titleBarStyle: "Overlay"`, which only macOS
  honors, so overlay mode is detected at runtime (`TITLEBAR_OVERLAY` in
  `src/lib/platform.ts`: inside Tauri, on a Mac); there is no build flag. In
  overlay mode the 44px app bar is the title bar:
  - It carries `data-tauri-drag-region="deep"`: every part of the bar that is
    not a link or a button drags the window, and a double-click zooms it. The
    controls stay clickable because Tauri never starts a drag on them.
  - It reserves 96px on the left for the traffic lights
    (`useTrafficLightInset`), and gives that space back in fullscreen, where
    macOS hides them.
  - `trafficLightPosition: { x: 15, y: 24 }` centers the lights in the bar. The
    `y` is not the distance from the top: tao sets the title bar container's
    height to the button height plus `y` and leaves the buttons where they sit
    inside it, so with macOS 26 and later metrics (14pt buttons, 9pt above the
    container's bottom edge) their center lands at `y - 2` = 22px. Earlier
    macOS versions have different title bar metrics, so the lights can sit a
    few pixels off center there.
- Shortcuts: Mod-K command palette, Mod-Enter run, Shift-Mod-Enter submit, `/`
  focuses library search.
- Coming Back to the library returns to where the list was scrolled
  (`useScrollMemory`); choosing another track starts from its top.

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
| `app/` | `AppShell`, `CommandPalette`, `AppLink`, `InlineButton`, `IconButton`, `ActionButton`, `CopyButton`, `ConfirmDialog`, `ShortcutHint`, `Page`, `PageHeader`, `Section`, `ScreenMessage`, `MetaList` / `MetaItem`, `RuledList`, `DisclosureChevron`, `Notice`, `EmptyState`, `ErrorState`, `LoadingLines`, `PixelIcon` |
| `prose/` | `Prose`, `CodeBlock` (both lazy; `CodeFrame` is the block they share with their fallbacks), `Figure`, `SvgFigureBody`, `ProblemFigures` |
| `editor/` | `CodeEditor` (with the completion list, documentation card, signature help, and diagnostics it draws; see below) |
| `problem/` | `DifficultyMark`, `StatusMark`, `JudgeList`, `FrameworkList`, `FrameworkSwitcher`, `ProgressRule`, `ProblemTable`, `ProblemRowList`, `ProblemHeader`, `PrerequisiteList`, `PrerequisiteNote`, `HintList`, `EditorialPanel`, `SubmissionList`, `SubmissionView` |
| `results/` | `ResultsPanel`, `VerdictHeader`, `VerdictMark`, `CaseMark`, `CaseList`, `MetricList`, `OutputBlock`, `RunProgress` |
| `library/` | `LibraryFilterBar`, `FilterSelect`, `TrackList`, `TrackSection` |
| `workbench/` | `StatementPane`, `EditorPane`, `PaneToolbar`, `EnvironmentNotice`, `HardwareNotice`, `SaveIndicator`, `SuggestionsStatus` |
| `models/` | `ModelsPanel` and `AssetRow` (the Models pane of Settings, built on `SettingRow`), `AssetStateMark`, `AssetProgress`, `AssetNotice` / `AssetNoticeList`, `AssetDownloadButton`, `RemoveAssetButton` |
| `targets/` | `TargetsPanel` and `TargetRow` (the Run targets pane of Settings, built on `SettingRow`), `TargetSummary`, `CheckStepList`, `CheckStepMark`, `FixText`, `AddTargetForm`, `TargetPicker` |
| `curriculum/` | `AreaEntry`, `AreaBanner`, `AreaSummary`, `ModulePlates`, `ModuleProgressBar`, `ModuleSwatch`, `ModuleHeader`, `ModuleNeighbors`, `ModuleRowList`, `SceneBanner` |
| `home/` | `ContinueEntry`, `HabitStrip`, `ActivityCalendar`, `GettingStarted`, `ProgressSummary`, `TrackProgress` |
| `settings/` | `SettingsNav`, `SettingList` / `SettingRow`, `RowStatus`, `SettingSwitch`, `PathRow`, `GeneralPanel`, `ThemePicker`, `GoalSetting`, `ReviewSetting`, `EnvironmentPanel`, `InstallButton`, `UvGuideButton`, `InstallLog`, `FilesPanel`, `CatalogIssueList` |

Patterns that recur, and the one component that owns each:

- Text that acts (a link in a sentence, a term with a tooltip): `InlineButton`;
  in-app navigation is `AppLink`, which is built on it.
- Rows separated by hairlines (submissions, cases, tracks, hints): `RuledList`.
- Problems in a reading order, as numbered lines that open: `ProblemRowList`.
- A titled block under an ink rule, on a page or inside a pane: `Section`.
- One line that needs attention, with its fix: `Notice` (tones warn, info,
  pass, fail).
- Nothing here, failed, or loading: `EmptyState`, `ErrorState`, `LoadingLines`;
  for a whole screen, inside `ScreenMessage`.
- A numbered, captioned drawing: `Figure`; an author's SVG inside it:
  `SvgFigureBody`. `ProblemFigures` wraps the prose of one problem so that
  `assets/...` images resolve through the backend; `Prose` itself knows nothing
  about problems.
- One thing in Settings on one line (its name, at most one line of help, its
  status, its control), with what follows under it and detail behind its name:
  `SettingRow`, inside a `SettingList`. Where such a row stands, as a shape and
  a word: `RowStatus`. A preference that is on or off: `SettingSwitch`, which
  is built on the row. The streamed log of an install, while it runs and after
  it failed: `InstallLog`.
- Copy to the clipboard: `CopyButton`. A row that opens: `Collapsible` with
  `DisclosureChevron`.

Home follows the first area the content defines. It opens on the scene of the
module the learner has reached (`SceneBanner`), with the suggested problem
under it (`ContinueEntry`, which names the module with its swatch: "Module III,
Transformer Forward Pass"). Then, top to bottom:

- `HabitStrip`: three facts under an ink rule, divided by hairlines. The streak
  with this week's seven days; the weekly goal (distinct problems solved since
  Monday against a number the learner sets under Settings, `GoalSetting`; the
  star appears when it is met); and reviews due, with a link that starts the
  first one. The reviews column is left out while reviews are turned off.
- "Due for review", when anything is due.
- "The route": a sentence of progress over the `ModuleProgressBar`
  (`AreaSummary`: "2 of 84 core problems solved."; one segment per module, as
  wide as its share of the core problems, filled in the module's color in
  whole-problem steps over a tinted track; a module with no core has no
  segment), and `ModulePlates` in its compact form: every module's scene in
  five columns with a pip per core problem, its roman numeral, and its name. A
  module whose core is complete carries the solved mark, and the star once
  every problem in it, core and further, is solved; the module that holds the
  next core problem says "Up next" beside its numeral and has its name in a
  heavier weight. A plate opens the module's page. The section's aside counts
  the modules whose core is complete.
- "Progress" (`ActivityCalendar`, then the counts by difficulty) beside "Recent
  submissions". The calendar shows accepted submits per day and appears only
  once the first of them is two weeks old, so a new learner is not shown an
  empty grid.

Until anything has been attempted, Home is a title page instead: the banner and
the first problem, the route, how a problem works (`GettingStarted`), and what
the modules cover (`TrackProgress` as contents). It never shows a page of zeros,
and it has no habit strip yet. The activity math (local days, Monday weeks) is
in `src/lib/activity.ts`.

The suggestion at the top of Home follows the route (`pickContinue` in
`src/lib/continue.ts`). Unfinished submitted work comes first ("Continue where
you left off"); otherwise it is the next unsolved core problem of the first
area, module by module, which for a new user is the first module's first core
problem ("Start here") and after a solve is the one that follows it ("Up
next"). For a core problem the entry says where it sits on the route
("Inference and serving, core 3 of 84") and links to the area. Once the core is
finished the suggestion goes on to the further problems in module order, and
after those to catalog order.

### The Curriculum screens

The catalog has one structure, and its words are the UI's words. An **area** is
a subject: an ordered list of modules. A **module** is a track: a scene, a
color, and its problems. A module's **core** problems are the ones every
learner of it should do, in teaching order; the rest are **further**. The
recommended route through an area is its core problems, module by module. The
model is `src/lib/curriculum.ts` (pure, built from `useAreas`, `useTracks`, and
`useProblems` by `useCurriculum`). When the content defines no areas, every
track forms one implicit area.

The Curriculum is its own destination in the app bar (between Home and
Library), not a section of the Library. The Library is a catalog: organized by
track, filtered, searched, windowed for hundreds of rows. The Curriculum is a
reading order, and as screens of its own it can be what it is: the table of
contents of the course. There are three screens, all in
`src/routes/curriculum.tsx`:

- **`/curriculum`**. With exactly one area this is that area's screen, with no
  extra hop. With several it lists them (`AreaEntry`): the sliced banner, the
  title, one sentence, the size in plain counts ("15 modules, 213 problems, 84
  core"), and the module bar; the whole block opens the area. Only real areas
  are listed: nothing is shown as planned.
- **`/curriculum/$area`**, the area. Its title and description head the page,
  then a sentence of progress with the next core problem linked (`AreaSummary`:
  "3 of 84 core problems solved. Next is ..., in Foundations."; before anything
  is solved it gives the length and the starting point instead of zeros) over
  the `ModuleProgressBar`. Under "Modules" are all of them as `ModulePlates`:
  what Home shows, plus a line of counts under each name ("6 core, 7 further").
  Modules that belong to no area follow under a quiet "More modules" heading as
  plain rows (`ModuleRowList`).
- **`/curriculum/$area/$module`**, the module's page. Its scene is the banner;
  under it the swatch and "Module III" with the way back to the area, the
  title, what the module covers, and one line of progress ("3 of 6 core solved,
  7 further"; `ModuleHeader`). Then a "Core" section and a "Further" section of
  `ProblemRowList`: the status mark, the number within the module (the further
  problems continue the core's count), the title in the reading serif
  (wrapping, never truncated), and the difficulty mark. The whole line opens
  the problem. A module with no core shows only "Further", and says so in the
  section's aside. At the foot are the previous and next modules, each with a
  small plate (`ModuleNeighbors`).

An unknown area or module id shows the empty state with a link to the
Curriculum.

What it deliberately is not: a roadmap. There are no nodes and connectors, no
locked steps, no percentages, no badges for finishing a module beyond the
solved mark and the star. The next problem is marked by the words "Up next" and
a medium weight, nothing else, on its row and on its module's plate. A
prerequisite that comes earlier in the area's reading order gets no remark,
because the order of the list already says so; one that comes later, or is
outside the area, gets the usual "best after ...".

The rest of the app points here. The workspace header names the problem's
module with its swatch and links to the module's page, and says in words
whether the problem is core ("Core 3 of 8") or further; the previous and next
buttons follow the reading order (a module's core, then its further problems,
then the next module). In the Library each track in the left list carries its
module swatch, and one filter, "Core", narrows the table to core problems or to
further ones.

### The Settings screen

Settings is a window of its own kind: a list of five categories at the left of
the page column and one pane beside it, never a long page. The categories are
General, Environment, Models, Run targets, and Files (`SettingsNav`). They are
links: the category is the `section` search parameter
(`routes.settings(section)`; General is the default and stays out of the URL;
an unknown value shows General), so a notice elsewhere in the app opens the
pane that fixes it. The list stays in place while a long pane scrolls, and the
current category is marked the way the Library marks its track.

A category carries a small warning mark when something in it needs the user:
an environment that cannot run a solution or an install that failed, a model
download that failed or stopped short, a run target whose last check failed,
catalog entries that could not be loaded. The decision is one pure function
(`settingsAttention` in `src/lib/settings.ts`). Things that are merely
optional and absent (MLX, a model never downloaded) are not marked.

Every pane is a `SettingList` of `SettingRow`s, and the rules are the row's:

- One row per thing, and each fact once. Python is one row of Environment; the
  language server (ty) is another, not a section of its own; the switch that
  turns suggestions on is a preference, so it is in General.
- A row is a name in the interface face, at most one short line of help under
  it and only where the name is not enough, and at the right its status
  (`RowStatus`: a shape and a word or a version) and its control. Statuses and
  controls share columns down the list, so they line up whatever their widths.
  Rows are divided by hairlines. A row is never a box: no background, no
  border of its own, no shadow.
- What is happening now goes under the row it belongs to and leaves when it is
  over: an install's log while it runs and after it failed, a download's
  progress, the reason a download or a check could not go on.
- Everything else is on demand, behind the row's name, which opens it: a
  model's source, path, and the problems that use it (the row itself says
  "used by 11 problems"); a target's machine, frameworks, and the steps of its
  last check. A target's details open by themselves while a check runs and
  whenever the last one failed, and close again when a check they opened for
  passes. "Add a machine" is one button; the form, or what to add to
  `~/.ssh/config` when there is no host to offer, appears when it is pressed.
- No section taglines and no paragraphs of explanation. A sentence that is
  still worth having lives where it matters: the Reviews row says what became
  of the schedules while reviews are off; that keys stay with SSH is said in
  the add-machine form; what removing deletes is said in the confirmation.

Conventions:

- A pattern used twice is a component. Look in the table above before writing
  markup in a route.
- Routes (`src/routes/`) contain layout and data wiring only. Every screen but
  Home is its own chunk: `src/router.tsx` defines each route's path and search
  validation, and the screen's module exports the rendering half as
  `Route = createLazyRoute(path)({ component })`, loaded with `.lazy()`.
  The three Curriculum screens share one module and one chunk; it exports a
  lazy route for each (`Route`, `AreaRoute`, `ModuleRoute`).
- In-app links are built from `src/lib/routes.ts` and rendered with `AppLink`.
- User-facing strings for enums live in `src/lib/labels.ts`; number, time, and
  metric formatting in `src/lib/format.ts`.
- Do not name a directory `workspace`: the repository's `.gitignore` ignores
  that name, and Tailwind skips ignored paths when scanning for classes.

### Known exceptions

- The search panel inside CodeMirror (Mod-F) is CodeMirror's own DOM, themed
  from the tokens. It is the one interactive surface not built from a shadcn
  primitive.
- What the language server adds to the editor (the completion list and its
  documentation panel, the hover card, signature help, diagnostic tooltips) is
  also CodeMirror's DOM: it has to be positioned and measured by the editor.
  It is themed in one place (`components/editor/language-theme.ts`) to read as
  the app's own layers: the popover surface, hairline, radius, and the one
  elevation; rows like the command palette's, with the kind of each
  suggestion written as a word ("function", "module") instead of an icon; a
  signature in the mono face on the listing tint over documentation in the
  interface face at 13px; a ruled heading per section ("Arguments",
  "Example"); math typeset by KaTeX. Cards cap their height at 22rem or 44% of
  the window and scroll inside. The active parameter in signature help is
  highlighted in straw, the color already reserved for highlights. A mistake
  is a thin wavy underline in the fail color with the word "Error" before its
  message; hints are not shown and nothing is added to the gutter.
- Generated primitives in `src/components/ui/` keep shadcn's own arbitrary
  values where they exist; app components use tokens only.

## 9. Authoring conventions the UI relies on

- Statements, hints, and editorials are Markdown with GFM tables, `$...$` and
  `$$...$$` math, and fenced code with a language tag (`python`, `json`,
  `bash`). Untagged fences render as plain listings.
- Callouts use GitHub alert syntax: `> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`.
- Use `##` for sections and `### Example N` for worked examples; the title is
  rendered by the app, so a statement should not start with an `#` heading.
