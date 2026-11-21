# Figures

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

- `box` (the default): `x`, `y`, `width`, `height`, `role`, `label`.
- `add`: an addition, a circle with a plus, centered `at` a point.
- `source`: a circle with a wave, for something that enters from the side, such as
  a position encoding: `at`, `role`, `label` (set above it), and `arrow_to`, the
  point its arrow ends at.

`role` is one of `attn`, `mlp`, `norm`, `embed`, `linear`, `softmax`, `cache`.

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
