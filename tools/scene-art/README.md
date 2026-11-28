# Scene art

Twenty-one illustrations, one per module, drawn as flat screenprints in one style.
Each area has its own series, a journey through one kind of landscape:

- "Inference and serving" (scenes 0 to 15): the mountains, from a cabin door before
  dawn and a trailhead at dawn to a summit observatory at night.
- "Model architectures" (scenes 16 to 20): the sea, from a harbor at dawn to a
  tall ship under the moon.

A module names its scene by slug in its `track.toml`. The places set a mood and
a color for each module; they do not depict the module's topic. A new area gets
a new landscape, so that its modules read as one set.

One scene is deliberately not a landscape. `trailblazer` shows a person, and its
prompt carries a modified style text (a hero figure instead of "no people"). That
text is inside its `prompt` field, so it will not match `style.txt`.

The shipped files are in `src/assets/scenes/`: `<slug>.webp` (1584x672, 33:14)
and `scenes.json`, an array in module order of `{ slug, accent }`, where
`accent` is the color measured from the finished image.

## Files

- `scenes.json`: for each scene, its position in module order (`order`), the slug, place, description, requested accent
  (name and hex), the scene and ink text, the full prompt that produced the
  shipped image, and an optional `crop` (`[left, top, right, bottom]` source
  pixels) that trims a poster margin the model sometimes bakes in, and an
  optional `printed` accent sampled by hand, for a scene whose lighter tint
  covers more area than its full accent (the measurement then picks the tint).
- `style.txt`: the shared prompt. Each scene's `prompt` is this text with
  `{scene}`, `{accent_name}`, `{accent_hex}`, and `{inks}` filled in. If you
  edit it, update the `prompt` fields to match.
- `generate.py`: sends prompts to the image model. Standard library only.
- `postprocess.py`: crops, resizes, writes WebP, measures accents, writes the
  UI manifest, and optionally a contact sheet. Needs Pillow.

## Regenerating

Raw generations and the API key stay out of the repo. Use a scratch folder.

```sh
export GEMINI_API_KEY=...            # never commit or print this
python3 generate.py --out-dir /path/to/raw --ref /path/to/style.jpg canyon
uv run --with pillow python postprocess.py /path/to/raw --contact /path/to/raw/contact.jpg
```

Omit the slug to generate every scene. `postprocess.py` takes slugs too
(`postprocess.py /path/to/raw harbor ship`): it then needs only those
`<slug>.jpg` files and leaves the other scenes' files and accents alone. With no
slugs it expects every scene's raw file.

The style reference used for the shipped set was the approved screenprint
probe, which is not stored here. Any finished scene, such as
`src/assets/scenes/trailhead.webp`, works as a stand-in for `--ref`.

## Reviewing a result

The model is not deterministic, so look at every result. Reject anything with
gradients, texture, outlines, text, people or animals, an accent that is not
dominant or has drifted in hue, or a landmark away from the center. About one
result in three comes back with an off-white poster margin: measure it and set
that scene's `crop`, and clear the `crop` of any scene whose new raw has none.
