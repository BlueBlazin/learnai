"""Check the SVG figures under `problems/` and the statements that reference them.

  python3 tools/figures/check.py [paths...]

With no path, every `problems/*/*/assets/*.svg` and the `statement.md`, `hints.md`,
and `editorial.md` beside them are checked. A path may be an SVG file, a Markdown
file, or a directory (a problem, or anything above it). One line is printed per
finding, `path: message`, and the exit status is 1 when there is an error.

`known.txt` lists findings (`path: message`, one per line) of figures drawn before
these rules. They are printed as `known: path: message` and do not fail the run.
Standard library only. The rules are in docs/PROBLEM_AUTHORING.md, "Figures".
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from maps import overlap, text_width

REPO = HERE.parent.parent
KNOWN = HERE / "known.txt"
REGISTRY = REPO / "src" / "components" / "figures" / "registry.ts"

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
TEXT_SIZE = 13
BASELINE = 4.5  # from the middle of a 13-unit line down to its baseline (as maps.py)
MAX_DESC_SENTENCES = 2
MAX_CAPTION_SENTENCES = 2

# ---------------------------------------------------------------- the policy
# Mirrors src/components/prose/svg-policy.ts. python/tests/test_figure_check.py
# fails when the two drift apart.

SVG_ELEMENT_NAMES = [
    "svg",
    "g",
    "defs",
    "title",
    "desc",
    "path",
    "line",
    "polyline",
    "polygon",
    "rect",
    "circle",
    "ellipse",
    "text",
    "tspan",
    "marker",
    "clipPath",
    "mask",
    "linearGradient",
    "radialGradient",
    "stop",
    "symbol",
    "use",
]

SVG_ELEMENTS = frozenset(SVG_ELEMENT_NAMES)

GEOMETRY = [
    "viewBox",
    "preserveAspectRatio",
    "width",
    "height",
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "d",
    "points",
    "dx",
    "dy",
    "rotate",
    "textLength",
    "lengthAdjust",
    "transform",
    "pathLength",
    "markerWidth",
    "markerHeight",
    "markerUnits",
    "refX",
    "refY",
    "orient",
    "clipPathUnits",
    "maskUnits",
    "maskContentUnits",
    "gradientUnits",
    "gradientTransform",
    "spreadMethod",
    "fx",
    "fy",
    "offset",
]

PRESENTATION = [
    "fill",
    "fill-opacity",
    "fill-rule",
    "stroke",
    "stroke-width",
    "stroke-opacity",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "color",
    "opacity",
    "stop-color",
    "stop-opacity",
    "clip-rule",
    "paint-order",
    "vector-effect",
    "shape-rendering",
    "font-family",
    "font-size",
    "font-style",
    "font-weight",
    "font-variant-numeric",
    "letter-spacing",
    "text-anchor",
    "dominant-baseline",
]

REFERENCING = ["clip-path", "mask", "marker-start", "marker-mid", "marker-end"]
ACCESSIBILITY = ["role", "aria-label", "aria-hidden"]

ATTRIBUTES = frozenset(GEOMETRY + PRESENTATION + REFERENCING + ACCESSIBILITY)
STYLE_PROPERTIES = frozenset(PRESENTATION)

ID = re.compile(r"^[A-Za-z_][\w.-]*$")
PLAIN_VALUE = re.compile(r"^[\w\s#%.,()+-]*$")
LOCAL_URL = re.compile(r"url\(\s*(['\"]?)#[A-Za-z_][\w.-]*\1\s*\)")
HREF_ELEMENTS = frozenset({"use", "linearGradient", "radialGradient"})
ALLOWED_COLOR = re.compile(r"^(none|currentColor|var\(\s*--figure-[a-z-]+\s*\))$")
STEP_ID = re.compile(r"^step-(\d+)$")

# ---------------------------------------------------------------- findings


class Finding:
    def __init__(self, path: str, message: str, error: bool = True) -> None:
        self.path = path
        self.message = message
        self.error = error

    def line(self) -> str:
        return f"{self.path}: {self.message}"


def shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def sentences(text: str) -> int:
    """How many sentences `text` has. A full stop followed by a lowercase word does not end one."""
    text = re.sub(r"\$[^$]*\$", "M", " ".join(text.split()))
    if not text:
        return 0
    return len(re.split(r"(?<=[.!?])\s+(?=[^a-z\s])", text))


def local(tag: str) -> str:
    """The name of an element without its namespace."""
    return tag.rsplit("}", 1)[-1]


def attribute_name(name: str) -> str:
    if name.startswith(f"{{{XLINK_NS}}}"):
        return "xlink:" + name.rsplit("}", 1)[-1]
    if name.startswith("{"):
        return name  # some other namespace: not kept
    return name


def style_of(element: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for declaration in element.get("style", "").split(";"):
        name, colon, value = declaration.partition(":")
        if colon and name.strip():
            result[name.strip().lower()] = value.strip()
    return result


# ---------------------------------------------------------------- one SVG


class SvgCheck:
    def __init__(self, path: Path) -> None:
        self.path = shown(path)
        self.found: list[Finding] = []
        self.seen: set[str] = set()
        self.source = path

    def add(self, message: str) -> None:
        if message not in self.seen:
            self.seen.add(message)
            self.found.append(Finding(self.path, message))

    def run(self) -> list[Finding]:
        try:
            root = ET.parse(self.source).getroot()
        except (ET.ParseError, OSError) as error:
            self.add(f"does not parse as XML: {error}")
            return self.found
        if root.tag != f"{{{SVG_NS}}}svg":
            self.add(
                f"the root is {local(root.tag)!r} in {root.tag.split('}')[0][1:]!r}, not an svg"
            )
            return self.found
        self.structure(root)
        self.policy(root)
        self.titles(root)
        self.stepped(root)
        self.lettering(root)
        return self.found

    # -- the root, titles, steps

    def structure(self, root: ET.Element) -> None:
        if "viewBox" not in root.attrib:
            self.add("the root has no viewBox")
        for name in ("width", "height"):
            if name in root.attrib:
                self.add(f"the root has a {name}; use a viewBox only")

    def titles(self, root: ET.Element) -> None:
        children = {local(child.tag): child for child in reversed(list(root))}
        if "title" not in children:
            self.add("the root has no <title>")
        desc = children.get("desc")
        if desc is None:
            self.add("the root has no <desc>")
        else:
            count = sentences("".join(desc.itertext()))
            if count > MAX_DESC_SENTENCES:
                self.add(
                    f"the root <desc> has {count} sentences, at most {MAX_DESC_SENTENCES}"
                )

    def stepped(self, root: ET.Element) -> None:
        steps: dict[int, ET.Element] = {}
        for child in root:
            identifier = child.get("id", "")
            if not identifier.startswith("step-"):
                continue
            match = STEP_ID.match(identifier)
            if local(child.tag) != "g" or match is None or int(match[1]) == 0:
                self.add(
                    f"{identifier!r} is not a step group (a <g> with id step-1, step-2, ...)"
                )
                continue
            number = int(match[1])
            if number in steps:
                self.add(f"the step id {identifier!r} is used twice")
            steps[number] = child
        if not steps:
            return
        missing = [n for n in range(1, max(steps) + 1) if n not in steps]
        if missing:
            self.add(
                "step ids must run from 1 without a gap: "
                + ", ".join(f"step-{n}" for n in missing)
                + " missing"
            )
        for number, group in sorted(steps.items()):
            first = next(iter(group), None)
            if first is None or local(first.tag) != "desc":
                self.add(f"step-{number}: the first child is not a <desc>")
                continue
            count = sentences("".join(first.itertext()))
            if count != 1:
                self.add(
                    f"step-{number}: the <desc> has {count} sentences, exactly 1 wanted"
                )

    # -- elements, attributes, style, colors

    def policy(self, root: ET.Element) -> None:
        for element in root.iter():
            if not isinstance(element.tag, str):
                continue
            name = local(element.tag)
            if element.tag != f"{{{SVG_NS}}}{name}" or name not in SVG_ELEMENTS:
                self.add(f"the element <{name}> is not kept by the app")
                continue
            for raw, value in element.attrib.items():
                attr = attribute_name(raw)
                self.attribute(name, attr, value)
            for prop, value in style_of(element).items():
                if prop not in STYLE_PROPERTIES:
                    self.add(f"the style property {prop!r} is not kept by the app")
                elif prop in ("fill", "stroke"):
                    self.color(prop, value)
                elif not PLAIN_VALUE.match(value) or re.search(
                    r"url|expression|image|attr\s*\(", value, re.IGNORECASE
                ):
                    self.add(f"the style value {prop}: {value} is not kept by the app")

    def attribute(self, element: str, name: str, value: str) -> None:
        if name == "id":
            if not ID.match(value):
                self.add(f"the id {value!r} is not kept by the app")
        elif name in ("href", "xlink:href"):
            target = value.strip()
            if (
                element not in HREF_ELEMENTS
                or not target.startswith("#")
                or not ID.match(target[1:])
            ):
                self.add(f"the {name} {value!r} is not kept by the app")
        elif name == "style":
            pass  # its properties are checked one by one
        elif name == "xmlns" or name.startswith("xmlns:"):
            pass
        elif name not in ATTRIBUTES:
            self.add(f"the attribute {name!r} on <{element}> is not kept by the app")
        elif name in ("fill", "stroke"):
            self.color(name, value)
        elif name != "aria-label":
            remainder = LOCAL_URL.sub("", value)
            if "url" in remainder.lower() or not PLAIN_VALUE.match(remainder):
                self.add(
                    f"the value of {name} on <{element}> ({value!r}) is not kept by the app"
                )

    def color(self, name: str, value: str) -> None:
        if not ALLOWED_COLOR.match(value.strip()):
            self.add(
                f"{name} is {value!r}; use none, currentColor, or var(--figure-...)"
            )

    # -- lettering

    def lettering(self, root: ET.Element) -> None:
        canvas = canvas_of(root)
        boxes: list[tuple[str, tuple[float, float, float, float], str | None]] = []

        def walk(
            element: ET.Element, chain: list[ET.Element], frame: str | None
        ) -> None:
            chain = chain + [element]
            if len(chain) == 2 and STEP_ID.match(element.get("id", "")):
                frame = element.get("id")
            if local(element.tag) == "text":
                self.text(element, chain, frame, canvas, boxes)
                return
            for child in element:
                if isinstance(child.tag, str):
                    walk(child, chain, frame)

        walk(root, [], None)
        for index, (label, box, frame) in enumerate(boxes):
            for other, other_box, other_frame in boxes[index + 1 :]:
                if (
                    frame is not None
                    and other_frame is not None
                    and frame != other_frame
                ):
                    continue
                if overlap(box, other_box):
                    self.add(f"the text boxes {label!r} and {other!r} overlap")

    def text(self, element, chain, frame, canvas, boxes) -> None:
        label = " ".join("".join(element.itertext()).split())
        size = font_size(element, chain)
        if size != TEXT_SIZE:
            shown_size = "not set" if size is None else f"{size:g}"
            self.add(f"the text {label!r} has font-size {shown_size}, not {TEXT_SIZE}")
        for child in element.iter():
            if child is not element and local(child.tag) == "tspan":
                child_size = font_size(child, chain + [child])
                if child_size != TEXT_SIZE and child_size != size:
                    self.add(
                        f"the tspan {' '.join(''.join(child.itertext()).split())!r} has font-size {child_size:g}, not {TEXT_SIZE}"
                    )
        offset = translation(chain)
        if offset is None or canvas is None:
            return  # moved by a transform this check cannot follow
        anchor = inherited(chain, "text-anchor") or "start"
        middle_line = inherited(chain, "dominant-baseline") in ("middle", "central")
        measure = size if size else TEXT_SIZE
        x0, y0 = first_number(element.get("x")), first_number(element.get("y"))
        x, y = (
            offset[0] + x0 + first_number(element.get("dx")),
            offset[1] + y0 + first_number(element.get("dy")),
        )
        for text_label, tx, ty in chunks(element, x, y, offset):
            if not text_label:
                continue
            width = text_width(text_label, measure)
            left = {"start": tx, "middle": tx - width / 2, "end": tx - width}.get(
                anchor, tx
            )
            middle = ty if middle_line else ty - BASELINE
            box = (left, middle - measure / 2, left + width, middle + measure / 2)
            if not (
                canvas[0] <= box[0]
                and canvas[1] <= box[1]
                and box[2] <= canvas[2]
                and box[3] <= canvas[3]
            ):
                self.add(f"the text {text_label!r} leaves the viewBox")
            boxes.append((text_label, box, frame))


def canvas_of(root: ET.Element) -> tuple[float, float, float, float] | None:
    numbers = re.findall(r"-?\d*\.?\d+(?:e-?\d+)?", root.get("viewBox", ""))
    if len(numbers) != 4:
        return None
    x, y, width, height = (float(n) for n in numbers)
    return (x, y, x + width, y + height)


def first_number(value: str | None) -> float:
    match = re.match(r"\s*(-?\d*\.?\d+)", value or "")
    return float(match[1]) if match else 0.0


def inherited(chain: list[ET.Element], name: str) -> str | None:
    """The value of a presentation property on the nearest element that sets it: style first."""
    for element in reversed(chain):
        value = style_of(element).get(name, element.get(name))
        if value is not None:
            return value.strip()
    return None


def font_size(element: ET.Element, chain: list[ET.Element]) -> float | None:
    chain = chain if chain and chain[-1] is element else chain + [element]
    value = inherited(chain, "font-size")
    match = re.fullmatch(r"(\d*\.?\d+)(px)?", value or "")
    return float(match[1]) if match else None


def translation(chain: list[ET.Element]) -> tuple[float, float] | None:
    """The sum of the translations above a text, or None when a transform does more than move."""
    dx = dy = 0.0
    for element in chain:
        transform = element.get("transform")
        if not transform:
            continue
        for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", transform):
            numbers = [float(n) for n in re.findall(r"-?\d*\.?\d+(?:e-?\d+)?", args)]
            if name != "translate" or not numbers:
                return None
            dx += numbers[0]
            dy += numbers[1] if len(numbers) > 1 else 0.0
    return (dx, dy)


def chunks(element: ET.Element, x: float, y: float, offset: tuple[float, float]):
    """The pieces of a text set at their own place: its own run, and each tspan that has an x or y."""
    own = [element.text or ""]
    placed = []
    for child in element:
        content = "".join(child.itertext())
        if local(child.tag) == "tspan" and ("x" in child.attrib or "y" in child.attrib):
            cx = offset[0] + first_number(child.get("x")) if "x" in child.attrib else x
            cy = offset[1] + first_number(child.get("y")) if "y" in child.attrib else y
            placed.append((" ".join(content.split()), cx, cy))
        else:
            own.append(content)
        own.append(child.tail or "")
    return [(" ".join("".join(own).split()), x, y), *placed]


# ---------------------------------------------------------------- statements

IMAGE = re.compile(r"!\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^)\s]*)[^)]*\)")
FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.DOTALL | re.MULTILINE)
TEXTS = ("statement.md", "hints.md", "editorial.md")


def interactive_names() -> set[str]:
    source = REGISTRY.read_text(encoding="utf-8") if REGISTRY.is_file() else ""
    return set(re.findall(r'"([a-z0-9]+(?:-[a-z0-9]+)*)"\s*:\s*lazy\(', source))


def check_texts(problem: Path, names: set[str]) -> tuple[list[Finding], set[str]]:
    """Findings in the Markdown of a problem, and the asset paths it references."""
    found: list[Finding] = []
    referenced: set[str] = set()
    for name in TEXTS:
        path = problem / name
        if not path.is_file():
            continue
        where = shown(path)
        seen: set[str] = set()

        def add(message: str, where: str = where, seen: set[str] = seen) -> None:
            if message not in seen:
                seen.add(message)
                found.append(Finding(where, message))

        text = FENCE.sub("", path.read_text(encoding="utf-8"))
        for caption, target in IMAGE.findall(text):
            if target.startswith("interactive:"):
                key = target[len("interactive:") :]
                if key not in names:
                    add(f"interactive:{key} is not a figure in the registry")
            elif target.startswith("assets/"):
                referenced.add(target)
                if not (problem / target).is_file():
                    add(f"{target} does not exist")
            else:
                continue
            count = sentences(caption)
            if count > MAX_CAPTION_SENTENCES:
                add(
                    f"the caption of {target} has {count} sentences, at most {MAX_CAPTION_SENTENCES}"
                )
    return found, referenced


# ---------------------------------------------------------------- running


def problem_of(path: Path) -> Path | None:
    for parent in [path, *path.parents]:
        if (parent / "problem.toml").is_file():
            return parent
    return None


def targets(paths: list[Path]) -> dict[Path, set[Path] | None]:
    """Problem directories to check, each with the SVGs asked for (None: every SVG it has)."""
    chosen: dict[Path, set[Path] | None] = {}
    if not paths:
        for toml in sorted((REPO / "problems").glob("*/*/problem.toml")):
            chosen[toml.parent] = None
        return chosen
    for path in paths:
        path = path.resolve()
        if path.is_dir():
            roots = (
                [problem_of(path)]
                if problem_of(path)
                else [t.parent for t in sorted(path.rglob("problem.toml"))]
            )
            for root in roots:
                chosen[root] = None
        elif (problem := problem_of(path)) is not None:
            if path.suffix == ".svg" and chosen.get(problem, set()) is not None:
                chosen.setdefault(problem, set()).add(path)  # type: ignore[union-attr]
            else:
                chosen.setdefault(problem, set())
    return chosen


def check_all(paths: list[Path]) -> list[Finding]:
    chosen = targets(paths)
    names = interactive_names()
    found: list[Finding] = []
    for problem, only in chosen.items():
        svgs = (
            sorted((problem / "assets").rglob("*.svg"))
            if (problem / "assets").is_dir()
            else []
        )
        for svg in svgs if only is None else sorted(only):
            found.extend(SvgCheck(svg).run())
        texts, referenced = check_texts(problem, names)
        found.extend(texts)
        for svg in svgs:
            if (only is None or svg in only) and svg.relative_to(
                problem
            ).as_posix() not in referenced:
                found.append(
                    Finding(
                        shown(svg),
                        "warning: no statement, hint, or editorial references this figure",
                        False,
                    )
                )
    return found


def read_known(path: Path) -> list[str]:
    if not path.is_file():
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check SVG figures and the statements that use them."
    )
    parser.add_argument(
        "paths", nargs="*", type=Path, help="SVG or Markdown files, or directories"
    )
    parser.add_argument(
        "--known", type=Path, default=KNOWN, help="allowlist of `path: message` lines"
    )
    args = parser.parse_args(argv)

    found = check_all(args.paths)
    known = read_known(args.known)
    failing = 0
    seen_known: set[str] = set()
    for finding in found:
        line = finding.line()
        if line in known:
            seen_known.add(line)
            print(f"known: {line}")
            continue
        print(line)
        failing += finding.error
    if not args.paths:
        for line in known:
            if line not in seen_known:
                print(
                    f"{shown(args.known)}: warning: no longer reported, remove it: {line}"
                )
    count = len(seen_known)
    print(
        f"figures: {failing} error{'' if failing == 1 else 's'}, {count} known",
        file=sys.stderr,
    )
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
