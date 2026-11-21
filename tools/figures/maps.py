"""Draw the architecture map of a module, once per problem that shows it.

A spec in `maps/<track>.json` describes the map of one module: blocks with a role
and a label, wires, the panel, and the parts each problem highlights. For every
problem the spec lists, this script writes `problems/<track>/<id>/assets/<file>`
with that problem's parts at full strength and the rest dimmed.

Run from the repository root (standard library only):

  python3 tools/figures/maps.py            # write every map
  python3 tools/figures/maps.py --check    # fail when a written file is out of date

See README.md for the spec format.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SPECS = HERE / "maps"

# The role fills of docs/DESIGN.md: `--figure-<role>`, one meaning each.
ROLES = ("attn", "mlp", "norm", "embed", "linear", "softmax", "cache")

TEXT = 13  # font size of every label, in drawing units
BASELINE = 4.5  # from the middle of a 13-unit line down to its baseline
PAD = 4  # least space between a label and the sides of its box
THIN = 1  # wires and whatever is not this problem's part
BOLD = 1.5  # this problem's part
DIM_FILL = 0.4
DIM_STROKE = 0.45
HEAD_LENGTH = 6.5
HEAD_HALF = 3.3
ADD_RADIUS = 8
SOURCE_RADIUS = 10
MARKER = "This problem"

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # left, top, right, bottom


class SpecError(Exception):
    """A spec that cannot be drawn, or whose labels do not fit."""


def num(value: float) -> str:
    """A coordinate rounded to one decimal, without a trailing `.0`."""
    rounded = round(float(value), 1)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


def text_width(text: str, size: float = TEXT) -> float:
    """An upper estimate of the width of `text` in the interface sans."""
    ems = 0.0
    for char in text:
        if char in "mwMW":
            ems += 0.9
        elif char in " il.,:;'|":
            ems += 0.35
        elif char.isupper() or char.isdigit() or char == "×":
            ems += 0.7
        else:
            ems += 0.58
    return ems * size


def text_box(text: str, x: float, middle: float, anchor: str = "middle") -> Box:
    """The box a label covers when it is set with its middle line at `middle`."""
    width = text_width(text)
    left = {"start": x, "middle": x - width / 2, "end": x - width}[anchor]
    return (left, middle - TEXT / 2, left + width, middle + TEXT / 2)


def overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------------- drawing


def text(
    label: str, x: float, middle: float, *, anchor: str = "middle", extra: str = ""
) -> str:
    return (
        f'<text x="{num(x)}" y="{num(middle + BASELINE)}" font-size="{TEXT}" '
        f'text-anchor="{anchor}"{extra}>{escape(label)}</text>'
    )


def head(tip: Point, direction: Point) -> str:
    """An arrowhead as a polygon with its tip at `tip`, pointing along `direction`."""
    (x, y), (dx, dy) = tip, direction
    bx, by = x - dx * HEAD_LENGTH, y - dy * HEAD_LENGTH
    corners = [
        (x, y),
        (bx - dy * HEAD_HALF, by + dx * HEAD_HALF),
        (bx + dy * HEAD_HALF, by - dx * HEAD_HALF),
    ]
    return " ".join(f"{num(px)},{num(py)}" for px, py in corners)


def unit(a: Point, b: Point) -> Point:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length == 0:
        raise SpecError(f"a wire repeats the point {a}")
    return ((b[0] - a[0]) / length, (b[1] - a[1]) / length)


def wire_path(points: list[Point], radius: float, arrow: bool) -> str:
    """The path of a wire: straight runs, corners rounded by `radius`, short of the arrowhead."""
    points = list(points)
    if arrow:
        dx, dy = unit(points[-2], points[-1])
        # The line stops inside the head, so its square end never shows past the tip.
        points[-1] = (
            points[-1][0] - dx * (HEAD_LENGTH - 1.5),
            points[-1][1] - dy * (HEAD_LENGTH - 1.5),
        )
    parts = [f"M{num(points[0][0])} {num(points[0][1])}"]
    for before, corner, after in zip(points, points[1:], points[2:], strict=False):
        inward, outward = unit(corner, before), unit(corner, after)
        parts.append(
            f"L{num(corner[0] + inward[0] * radius)} {num(corner[1] + inward[1] * radius)}"
        )
        parts.append(
            f"Q{num(corner[0])} {num(corner[1])} "
            f"{num(corner[0] + outward[0] * radius)} {num(corner[1] + outward[1] * radius)}"
        )
    parts.append(f"L{num(points[-1][0])} {num(points[-1][1])}")
    return " ".join(parts)


def paint(role: str | None, strong: bool, dimmed: bool) -> str:
    """The fill and outline attributes of one block shape."""
    attrs = f' stroke="currentColor" stroke-width="{num(BOLD if strong else THIN)}"'
    if dimmed:
        attrs += f' stroke-opacity="{DIM_STROKE}"'
        if role:
            attrs += f' fill-opacity="{DIM_FILL}"'
    if role:
        return f'{attrs} style="fill: var(--figure-{role})"'
    return f' fill="none"{attrs}'


def draw_block(block: dict, strong: bool, dimmed: bool) -> list[str]:
    shape = block.get("shape", "box")
    muted = ' style="fill: var(--figure-muted)"' if dimmed else ""
    if shape == "box":
        x, y, width, height = block["x"], block["y"], block["width"], block["height"]
        return [
            (
                f'<rect x="{num(x)}" y="{num(y)}" width="{num(width)}" height="{num(height)}" rx="5"'
                f"{paint(block.get('role'), strong, dimmed)}/>"
            ),
            text(block["label"], x + width / 2, y + height / 2, extra=muted),
        ]
    cx, cy = block["at"]
    if shape == "add":
        arm = ADD_RADIUS - 3.5
        return [
            f'<circle cx="{num(cx)}" cy="{num(cy)}" r="{ADD_RADIUS}"{paint(None, strong, dimmed)}/>',
            (
                f'<path d="M{num(cx - arm)} {num(cy)} H{num(cx + arm)} M{num(cx)} {num(cy - arm)} V{num(cy + arm)}"'
                f"{paint(None, strong, dimmed)}/>"
            ),
        ]
    # A source: a circle with a wave in it, its name above, and an arrow into the block it feeds.
    radius = SOURCE_RADIUS
    tip = tuple(block["arrow_to"])
    direction = unit((cx, cy), tip)
    start = (cx + direction[0] * radius, cy + direction[1] * radius)
    opacity = f' fill-opacity="{DIM_STROKE}"' if dimmed else ""
    return [
        f'<path d="{wire_path([start, tip], 0, True)}"{paint(None, strong, dimmed)}/>',
        f'<polygon points="{head(tip, direction)}"{opacity}/>',
        f'<circle cx="{num(cx)}" cy="{num(cy)}" r="{radius}"{paint(block["role"], strong, dimmed)}/>',
        f'<path d="M{num(cx - 6.5)} {num(cy)} q3.25 -8 6.5 0 t6.5 0"{paint(None, strong, dimmed)}/>',
        text(block["label"], cx, cy - radius - 9, extra=muted),
    ]


def render(spec: dict, problem: str) -> str:
    """The SVG of the map as `problem` shows it."""
    focus = spec["problems"][problem]
    parts = focus.get("parts")
    on = set(parts or [])

    def strong(item: dict) -> bool:
        return parts is None or item.get("part") in on

    def dimmed(item: dict) -> bool:
        return parts is not None and item.get("part") not in on

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {num(spec["width"])} {num(spec["height"])}">',
        f"  <title>{escape(spec['title'])}</title>",
        f"  <desc>{escape(' '.join(filter(None, [spec['desc'], focus.get('desc')])))}</desc>",
        (
            f"  <!-- Written by tools/figures/maps.py from maps/{spec['track']}.json for"
            f" {'/'.join(place(spec, problem))}. Change the spec and run the script; do not edit this file. -->"
        ),
    ]

    panel = spec.get("panel")
    if panel:
        out.append("")
        out.append("  <!-- The panel: the group of blocks that repeats -->")
        outline = (
            f' stroke="currentColor" stroke-width="{num(BOLD)}" style="fill: var(--figure-panel)"'
            if parts is not None and strong(panel)
            else f' stroke-width="{num(THIN)}" style="fill: var(--figure-panel); stroke: var(--figure-faint)"'
        )
        out.append(
            f'  <rect x="{num(panel["x"])}" y="{num(panel["y"])}" width="{num(panel["width"])}" '
            f'height="{num(panel["height"])}" rx="10"{outline}/>'
        )
        if panel.get("label"):
            out.append("  " + text(panel["label"], *panel["label_at"]))

    wires = spec.get("wires", [])
    for title, width, chosen in (
        ("Wires", THIN, [w for w in wires if parts is None or not strong(w)]),
        (
            "Wires of this problem",
            BOLD,
            [w for w in wires if parts is not None and strong(w)],
        ),
    ):
        if not chosen:
            continue
        out.append("")
        out.append(f"  <!-- {title} -->")
        out.append(
            f'  <g fill="none" stroke="currentColor" stroke-width="{num(width)}">'
        )
        for wire in chosen:
            points = [tuple(point) for point in wire["points"]]
            out.append(
                f'    <path d="{wire_path(points, wire.get("round", 0), wire.get("arrow", False))}"/>'
            )
        out.append("  </g>")
    marks = []
    for wire in wires:
        points = [tuple(point) for point in wire["points"]]
        if wire.get("arrow"):
            marks.append(
                f'<polygon points="{head(points[-1], unit(points[-2], points[-1]))}"/>'
            )
        if wire.get("tap"):
            marks.append(
                f'<circle cx="{num(points[0][0])}" cy="{num(points[0][1])}" r="2.5"/>'
            )
    if marks:
        out.append("  <!-- Arrowheads, and a dot where a wire leaves another -->")
        out.append("  <g>")
        out.extend(f"    {mark}" for mark in marks)
        out.append("  </g>")

    out.append("")
    out.append("  <!-- Blocks, top to bottom -->")
    for block in spec["blocks"]:
        state = (
            "this problem"
            if parts is not None and strong(block)
            else "dimmed"
            if dimmed(block)
            else "shown"
        )
        out.append(f"  <g><!-- {block['part']}: {state} -->")
        out.extend(
            f"    {line}"
            for line in draw_block(
                block, parts is not None and strong(block), dimmed(block)
            )
        )
        out.append("  </g>")

    labels = spec.get("labels", [])
    if labels:
        out.append("")
        out.append("  <!-- What goes in and what comes out -->")
        out.append('  <g style="fill: var(--figure-muted)">')
        out.extend(f"    {text(label['text'], *label['at'])}" for label in labels)
        out.append("  </g>")

    marker = focus.get("marker")
    if marker:
        x, y = marker["from"]
        end = x + marker["length"]
        out.append("")
        out.append("  <!-- The marker of this problem's part -->")
        out.append(
            f'  <line x1="{num(x)}" y1="{num(y)}" x2="{num(end)}" y2="{num(y)}" fill="none" '
            f'stroke="currentColor" stroke-width="{num(BOLD)}"/>'
        )
        out.append(
            "  " + text(MARKER, end + 6, y, anchor="start", extra=' font-weight="600"')
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- checks


def place(spec: dict, key: str) -> tuple[str, str]:
    """The (track, id) of a key under `problems`: a bare id is in the spec's own track."""
    parts = key.split("/")
    if len(parts) == 1:
        parts = [spec["track"], key]
    if len(parts) != 2 or not all(parts):
        raise SpecError(
            f"{spec['track']}.json: the problem {key!r} must be an id or track/id"
        )
    return parts[0], parts[1]


def block_box(block: dict) -> Box:
    shape = block.get("shape", "box")
    if shape == "box":
        return (
            block["x"],
            block["y"],
            block["x"] + block["width"],
            block["y"] + block["height"],
        )
    radius = ADD_RADIUS if shape == "add" else SOURCE_RADIUS
    cx, cy = block["at"]
    return (cx - radius, cy - radius, cx + radius, cy + radius)


def problems_of(spec: dict) -> list[str]:
    """Why `spec` cannot be drawn: unknown roles and parts, and labels that do not fit."""
    found: list[str] = []
    names = {item["part"] for item in spec["blocks"]}
    names |= {wire["part"] for wire in spec.get("wires", []) if "part" in wire}
    if spec.get("panel"):
        names.add(spec["panel"]["part"])
    canvas: Box = (0, 0, spec["width"], spec["height"])

    # Every label with the box it covers, and the block it belongs to (if any).
    lettering: list[tuple[str, Box, str | None]] = []
    for block in spec["blocks"]:
        shape = block.get("shape", "box")
        if shape not in ("box", "add", "source"):
            found.append(f"block {block['part']}: unknown shape {shape!r}")
            continue
        # A box may have no role (drawn unfilled); a source needs one.
        role = block.get("role")
        if role not in ROLES and (
            shape == "source" or (shape == "box" and role is not None)
        ):
            found.append(f"block {block['part']}: unknown role {role!r}")
        if shape == "box":
            middle = (block["x"] + block["width"] / 2, block["y"] + block["height"] / 2)
            lettering.append(
                (block["label"], text_box(block["label"], *middle), block["part"])
            )
            if (
                text_width(block["label"]) + 2 * PAD > block["width"]
                or TEXT + 2 > block["height"]
            ):
                found.append(
                    f"block {block['part']}: the label {block['label']!r} does not fit its box"
                )
        elif shape == "source":
            cx, cy = block["at"]
            lettering.append(
                (
                    block["label"],
                    text_box(block["label"], cx, cy - SOURCE_RADIUS - 9),
                    None,
                )
            )
    panel = spec.get("panel")
    if panel and panel.get("label"):
        lettering.append(
            (panel["label"], text_box(panel["label"], *panel["label_at"]), None)
        )
    for label in spec.get("labels", []):
        lettering.append((label["text"], text_box(label["text"], *label["at"]), None))

    def check(labels: list[tuple[str, Box, str | None]], where: str) -> None:
        for index, (label, box, owner) in enumerate(labels):
            if not (
                canvas[0] <= box[0]
                and canvas[1] <= box[1]
                and box[2] <= canvas[2]
                and box[3] <= canvas[3]
            ):
                found.append(f"{where}: the label {label!r} leaves the drawing")
            for other, other_box, _ in labels[index + 1 :]:
                if overlap(box, other_box):
                    found.append(f"{where}: the labels {label!r} and {other!r} overlap")
            for block in spec["blocks"]:
                if block["part"] != owner and overlap(box, block_box(block)):
                    found.append(
                        f"{where}: the label {label!r} overlaps the block {block['part']}"
                    )

    if not spec["problems"]:
        check(lettering, spec["track"])
    for problem, focus in spec["problems"].items():
        where = "/".join(place(spec, problem))
        for part in focus.get("parts") or []:
            if part not in names:
                found.append(f"{where}: no part named {part!r}")
        marker = focus.get("marker")
        extra = []
        if marker:
            x, y = marker["from"]
            extra.append(
                (MARKER, text_box(MARKER, x + marker["length"] + 6, y, "start"), None)
            )
        check(lettering + extra, where)
    return found


# ---------------------------------------------------------------- files


def load_specs(paths: list[Path]) -> list[dict]:
    specs = []
    for path in sorted(paths):
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("track") != path.stem:
            raise SpecError(
                f'{path.name}: "track" must be {path.stem!r}, the name of the file'
            )
        found = problems_of(spec)
        if found:
            raise SpecError(f"{path.name}:\n  " + "\n  ".join(found))
        specs.append(spec)
    return specs


def outputs(specs: list[dict], root: Path) -> dict[Path, str]:
    """Every file the specs describe, with its content."""
    files: dict[Path, str] = {}
    for spec in specs:
        for problem in spec["problems"]:
            track, name = place(spec, problem)
            directory = root / "problems" / track / name
            if not (directory / "problem.toml").is_file():
                raise SpecError(
                    f"{spec['track']}.json: there is no problem {track}/{name}"
                )
            files[directory / "assets" / spec["file"]] = render(spec, problem)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the architecture maps of the modules."
    )
    parser.add_argument(
        "specs", nargs="*", type=Path, help="spec files (default: every maps/*.json)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; fail when a file is out of date",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO,
        help="the repository root (default: this checkout)",
    )
    args = parser.parse_args(argv)

    try:
        files = outputs(load_specs(args.specs or list(SPECS.glob("*.json"))), args.root)
    except SpecError as error:
        print(f"maps: {error}", file=sys.stderr)
        return 2

    stale = 0
    for path, content in files.items():
        shown = path.relative_to(args.root)
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == content:
            continue
        stale += 1
        if args.check:
            print(
                f"{'out of date' if current is not None else 'missing'}: {shown}",
                file=sys.stderr,
            )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"wrote {shown}")
    if args.check and stale:
        print(
            "maps: run `python3 tools/figures/maps.py` and commit the result",
            file=sys.stderr,
        )
        return 1
    if not stale:
        print(f"maps: {len(files)} file{'' if len(files) == 1 else 's'} up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
