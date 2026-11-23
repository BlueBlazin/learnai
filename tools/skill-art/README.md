# Skill art

Sixteen graphics, one per skill of the Accomplishments screen, drawn as flat
screenprints in the style of the scenes (`tools/scene-art/`). Each shows the
same traveler performing one action: setting out on a trail, looking through a
spyglass, crossing a rope bridge. The action loosely echoes the skill and does
not depict its topic.

The traveler wears a wide-brimmed hat and a knee-length coat and carries a
rucksack with a blanket roll. The figure is not a black cutout: every part is
built from two or three flat tones that show a lit side and a shadow side, and
the pack has its straps, flap, and pockets. The face is a plain shape in one
skin tone with no features.

Each print takes one accent color beside the three inks every print shares
(off-white paper, blue-black, steel blue). A skill names its graphic by slug in
the `art` field of its `skills/<id>.toml`, and its `color` is the accent of
that print.

The shipped files are in `src/assets/skills/`: `<slug>.webp`, 768 px square.

## Files

- `skills.json`: one entry per graphic, plus `wanderer`, the character study.
  Each has the slug, a `label` (the skill's title and name, printed on the
  contact sheet), the accent (name and hex), and the `scene` and `inks` text.
- `style.txt`: the shared prompt, with `{refs}`, `{scene}`, `{accent_name}`,
  `{accent_hex}`, and `{inks}` filled in per graphic. `{refs}` is the sentence
  that says what the attached images are; `generate.py` supplies it.
- `wanderer.jpg`: the approved character study. It is passed as
  `--character-ref` so that every print draws the same traveler.
- `generate.py`: sends prompts to the image model. Standard library only.
- `postprocess.py`: trims a margin, crops to a square, resizes, writes WebP,
  and optionally a contact sheet. Needs Pillow.

## Regenerating

The key is read from `GEMINI_API_KEY` only. Raw generations and the key stay
out of the repo. Use a scratch folder.

```sh
export GEMINI_API_KEY=...            # never commit or print this
python3 generate.py --out-dir /path/to/raw \
    --style-ref ../../src/assets/scenes/trailhead.webp \
    --character-ref wanderer.jpg spyglass
uv run --with pillow python postprocess.py /path/to/raw --contact /path/to/raw/contact.jpg
```

`generate.py`:

- Omit the slug to generate every skill graphic. The `wanderer` study is
  generated only when named, and it is sent without the character reference.
- `--style-ref` (required) is a finished scene; the shipped set used
  `src/assets/scenes/trailhead.webp`. `--character-ref` is optional; without it
  the prompt describes the traveler in words only.
- `--variants N` makes N takes per slug, written as `<slug>-<n>.jpg` instead of
  `<slug>.jpg`. `postprocess.py` reads `<slug>.jpg`, so rename the take you
  keep.
- `--jobs N` is the number of requests in flight at once (default 4).

`postprocess.py` takes slugs too (`postprocess.py /path/to/raw spyglass
uphill`) and then writes only those files. A slug with no raw file is reported
as missing and skipped. `--out-dir` writes somewhere other than
`src/assets/skills/`. The `wanderer` study is never written to the app.

## Reviewing a result

The model is not deterministic, so look at every result, at full size and at
the size of a tile (124 px). Reject:

- a face that is white or off-white, or that has eyes, a mouth, or any other
  feature;
- outlines or line work around the figure or anywhere else;
- a pose that reads wrong at tile size, so that the action cannot be told;
- gradients, texture, text, a second person, or an animal;
- an accent that is not dominant or has drifted in hue.

Some results come back with an off-white poster margin baked in. That one is
not a reason to reject: `postprocess.py` finds a uniform paper border and trims
it before cropping.

## The shipped set

Three of the sixteen are second takes: `spyglass`, `uphill`, and `campfire`.
They were made after `style.txt` gained the "never white" wording for the
face, and after the `uphill` and `campfire` scene texts were tightened. The
other thirteen were made just before those changes, so regenerating one of
them now uses a slightly different style text than the one that produced the
shipped file.

## Adding a graphic

Add an entry to `skills.json` with a new slug, an accent no other print uses,
and `scene` and `inks` text in the manner of the others: one action, seen from
the side, with the traveler's outline against a lighter area. Generate it with
both references, review it, run `postprocess.py` for that slug, and set `art`
and `color` in the skill's file.
