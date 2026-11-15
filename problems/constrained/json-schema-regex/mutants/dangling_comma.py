# BUG: every property but the last is followed by a comma, and an optional property is wrapped in (...)? together with that comma, so leaving out the last property leaves a dangling comma and the valid document without it is rejected.
import json

INTEGER = r"-?(0|[1-9][0-9]*)"
NUMBER = INTEGER + r"(\.[0-9]+)?([eE][+\-]?[0-9]+)?"
# One item of a string: an ordinary character (anything but the quote, the backslash,
# and the control characters), a two-character escape, or \u and four hex digits.
STRING_ITEM = r'([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'
METACHARACTERS = set(".^$*+?{}[]()|\\")


def literal(text: str) -> str:
    """A pattern that matches exactly `text`: every metacharacter gets a backslash."""
    return "".join("\\" + char if char in METACHARACTERS else char for char in text)


def string(schema: dict) -> str:
    low = schema.get("minLength", 0)
    high = schema.get("maxLength")
    if high is None:
        count = "*" if low == 0 else f"{{{low},}}"
    else:
        count = f"{{{low},{high}}}"
    return f'"{STRING_ITEM}{count}"'


def array(schema: dict) -> str:
    item = schema_to_regex(schema["items"])
    low = schema.get("minItems", 0)
    high = schema.get("maxItems")
    if high == 0:
        return r"\[\]"
    # A non-empty array is one item and then (comma, item) repeated, so the comma count
    # is one less than the item count and no comma is ever left dangling.
    more = max(low, 1) - 1
    count = f"{{{more},}}" if high is None else f"{{{more},{high - 1}}}"
    body = f"{item}(,{item}){count}"
    return rf"\[{body}\]" if low >= 1 else rf"\[({body})?\]"


def obj(schema: dict) -> str:
    required = set(schema.get("required", ()))
    members = [
        (literal(json.dumps(name)) + ":" + schema_to_regex(sub), name in required)
        for name, sub in schema.get("properties", {}).items()
    ]

    pieces = []
    for i, (member, needed) in enumerate(members):
        piece = member + ("," if i < len(members) - 1 else "")
        pieces.append(piece if needed else f"({piece})?")
    return r"\{" + "".join(pieces) + r"\}"


def schema_to_regex(schema: dict) -> str:
    if "enum" in schema:
        return "(" + "|".join(literal(json.dumps(value)) for value in schema["enum"]) + ")"
    kind = schema.get("type")
    if kind == "null":
        return "null"
    if kind == "boolean":
        return "(true|false)"
    if kind == "integer":
        return INTEGER
    if kind == "number":
        return NUMBER
    if kind == "string":
        return string(schema)
    if kind == "array":
        return array(schema)
    if kind == "object":
        return obj(schema)
    raise ValueError(f"unsupported schema type: {kind!r}")
