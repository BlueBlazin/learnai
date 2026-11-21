# Figures

Two scripts live here: `maps.py` draws the architecture maps, and `check.py` checks
every figure (see "Checking figures" below).

`maps.py` draws the architecture map of a module: one drawing of the whole model,
written once per problem with that problem's part at full strength and the rest
dimmed. The rules every figure follows, and the other kinds of figure, are in
`docs/PROBLEM_AUTHORING.md` under "Figures".

## Running

From the repository root, with the standard library only:

```sh
python3 tools/figures/maps.py            # write every map
python3 tools/figures/maps.py --check    # write nothing; exit 1 when a file is out of date
```

Each problem a spec lists gets `problems/<track>/<id>/assets/<file>`. The output
depends only on the spec, so running the script twice changes nothing. Commit the
written files. `python/tests/test_figure_maps.py` runs `--check`.

The script refuses a spec whose labels do not fit: a label wider than its box, two
labels that overlap, a label over another block, or a label outside the drawing.
Widths are an upper estimate for 13-unit text.

## The spec

`maps/<track>.json` describes one module. Coordinates are drawing units, `y` down.

| key | meaning |
|---|---|
| `track` | the module; must match the file name |
| `file` | the name of the written file in each problem's `assets/` |
| `title`, `desc` | the `<title>`, and the first sentence of the `<desc>` |
| `width`, `height` | the `viewBox`; the width is 480 |
| `panel` | the ground behind the blocks that repeat: `part`, `x`, `y`, `width`, `height`, and an optional `label` at `label_at` |
| `blocks` | the blocks, top to bottom (see below) |
| `wires` | each has `points`; optional `arrow` (a head at the last point), `tap` (a dot at the first), `round` (corner radius), and `part` |
| `labels` | muted text outside the blocks: `text` and `at`, its middle |
| `problems` | per problem id: `parts`, `marker`, `desc` |

A block has a `part` (its name in this spec) and a `shape`:

- `box` (the default): `x`, `y`, `width`, `height`, `label`, and an optional `role`.
  A box without a `role` is drawn unfilled (`fill="none"`), and dimmed like the others.
- `add`: an addition, a circle with a plus, centered `at` a point.
- `source`: a circle with a wave, for something that enters from the side, such as
  a position encoding: `at`, `role`, `label` (set above it), and `arrow_to`, the
  point its arrow ends at.

`role` is one of `attn`, `mlp`, `norm`, `embed`, `linear`, `softmax`, `cache`.
A block that has no role of its own, such as a plain "Input" box, leaves it out.

A key under `problems` is a bare id, a problem of the spec's own track, or
`track/id`, a problem of another module: the map is written into that problem's
`assets/`, so a module can show a map in a problem that belongs to another one.

For a problem, `parts` lists what it highlights: blocks, the panel, and wires that
carry a `part`. Everything else is dimmed; wires are never dimmed, only drawn
thinner. `"parts": null` highlights nothing and dims nothing, for a problem that
is the whole model. `marker` draws a short line from `from`, `length` units to
the right, and "This problem" after it. `desc` is one sentence that says what is
highlighted; it becomes the second sentence of the `<desc>`.

## Adding a problem

Add its id under `problems` in the module's spec, run the script, and reference
the file from the statement:

```markdown
![A decoder-only transformer, token ids in at the bottom. This problem is ...](assets/map.svg)
```

## Checking figures

```sh
python3 tools/figures/check.py               # every problems/*/*/assets/*.svg and the texts that use them
python3 tools/figures/check.py problems/transformer/rope   # one problem, or an SVG or Markdown file
```

One line per finding, `path: message`; the exit status is 1 when there is an error.
It checks the rules of `docs/PROBLEM_AUTHORING.md` ("Figures"): the file parses with a
`viewBox` and no `width` or `height`; only elements, attributes, and style properties
that `src/components/prose/svg-policy.ts` keeps (the lists are mirrored in the script,
and `python/tests/test_figure_check.py` fails when they drift apart); no fixed color;
a `<title>` and a `<desc>` of at most two sentences; every text at `font-size` 13; no
text outside the `viewBox` and no two texts overlapping (an estimate, as in `maps.py`;
a stepped figure compares one frame at a time); stepped figures with ids from
`step-1` and a one-sentence `<desc>` first in each; and, in `statement.md`,
`hints.md`, and `editorial.md`, figures that exist, interactive names that are in
`src/components/figures/registry.ts`, and captions of two sentences at most. An SVG
that nothing references is a warning.

`known.txt` lists findings of figures drawn before these rules, as `path: message`
lines. They are printed as `known: ...` and do not fail the run, so a new finding
stands out. Fix a figure, then delete its lines; a line that no longer applies is
flagged. To see everything again, run with `--known /dev/null`.

To look at the figures in the app, run `pnpm dev` and open `#/dev/figures` (the
"Figures" link in the app bar). It shows the figures of every statement as a
statement draws them; `?track=<module>` (or the control at the top) limits it to one
module.
