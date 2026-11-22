# App icon

A snow-capped summit with a pennant on its tip, under a cyan sky: one more print
in the manner of the scenes (`tools/scene-art/`), simplified until it reads at
16 px.

## Files

- `art.jpg`: the square artwork as the image model returned it (1024x1024, with
  a red sky).
- `prompt.txt`: the prompt that produced it. It is the scene-art style text,
  changed for a square picture with one centered subject and very few shapes.
- `build.py`: reprints the red sky in cyan, cuts the art to the macOS icon
  shape, and writes the master. Needs Pillow and NumPy.
- `app-icon.png`: the master, 1024x1024, the source for `tauri icon`.

The shipped files are in `src-tauri/icons/`.

## Rebuilding

```sh
uv run --with pillow --with numpy python tools/app-icon/build.py
pnpm tauri icon tools/app-icon/app-icon.png -o /path/to/scratch
```

`tauri icon` also writes Android and iOS sets and a `64x64.png`, which this app
does not ship, so send its output to a scratch folder and copy over only the
files that `src-tauri/icons/` already holds.

To change the sky, edit `SKY` in `build.py`. The horizon band is a tint of the
same ink and follows it. The other inks are left as drawn.

## Redrawing

`art.jpg` was made with the same model and call as the scenes (see
`tools/scene-art/generate.py`), with `aspectRatio` set to `1:1`,
`src/assets/scenes/trailhead.webp` as the style reference, and `prompt.txt` as
the text. The model is not deterministic: a first take with a two-tone triangle
for the peak came back as a sailboat, which is why the prompt says what the
mountain must not look like. If the art is redrawn, measure its inks again and
update `MEASURED` in `build.py`.
