# BUG: json_complete treats every number state as complete, so texts that stop inside a number ("-", "1.", "2e", "2e+") count as complete documents.
class Invalid(ValueError):
    """Raised by json_step when the text can no longer become valid JSON."""


# A state is the tuple (mode, stack, rest):
#   mode   what the automaton expects next (the finite control),
#   stack  the open containers, outermost first, as a string of "{" and "[",
#   rest   mode-specific data: for a literal the characters still to come, inside a
#          string whether it is an object key, inside \uXXXX also the digits left.
# All three are immutable, so every state stays valid after it has been stepped.

WHITESPACE = " \t\n\r"
DIGITS = "0123456789"
HEX_DIGITS = "0123456789abcdefABCDEF"
SIMPLE_ESCAPES = '"\\/bfnrt'
LITERALS = {"t": "rue", "f": "alse", "n": "ull"}
# Number modes in which the number read so far is complete. A number has no closing
# character: it ends when a character arrives that cannot continue it.
NUMBER_DONE = ("zero", "int", "frac", "exp")


def json_start():
    return ("value", "", None)


def json_complete(state) -> bool:
    mode, stack, _ = state
    in_number = mode in NUMBER_DONE or mode in ("minus", "dot", "exp_mark", "exp_sign")
    return not stack and (mode == "after" or in_number)


def json_step(state, ch: str):
    mode, stack, rest = state

    # -- inside a string -------------------------------------------------------------
    if mode == "string":
        if ch == '"':
            return ("colon" if rest else "after", stack, None)
        if ch == "\\":
            return ("escape", stack, rest)
        if ch < " ":  # U+0000 to U+001F must be escaped
            raise Invalid(f"control character {ch!r} in a string")
        return state
    if mode == "escape":
        if ch == "u":
            return ("hex", stack, (rest, 4))
        if ch in SIMPLE_ESCAPES:
            return ("string", stack, rest)
        raise Invalid(f"invalid escape \\{ch}")
    if mode == "hex":
        is_key, left = rest
        if ch not in HEX_DIGITS:
            raise Invalid(f"{ch!r} is not a hex digit")
        return ("string", stack, is_key) if left == 1 else ("hex", stack, (is_key, left - 1))

    # -- inside true, false, null ----------------------------------------------------
    if mode == "literal":
        if ch != rest[0]:
            raise Invalid(f"expected {rest[0]!r}, got {ch!r}")
        return ("literal", stack, rest[1:]) if len(rest) > 1 else ("after", stack, None)

    # -- inside a number -------------------------------------------------------------
    if mode == "minus":
        if ch == "0":
            return ("zero", stack, None)
        if ch in DIGITS:
            return ("int", stack, None)
        raise Invalid(f"expected a digit after '-', got {ch!r}")
    if mode == "int" and ch in DIGITS:  # not in mode "zero": no digit may follow a leading 0
        return state
    if mode in ("zero", "int") and ch == ".":
        return ("dot", stack, None)
    if mode in ("zero", "int", "frac") and ch in "eE":
        return ("exp_mark", stack, None)
    if mode in ("dot", "frac") and ch in DIGITS:
        return ("frac", stack, None)
    if mode == "exp_mark" and ch in "+-":
        return ("exp_sign", stack, None)
    if mode in ("exp_mark", "exp_sign", "exp") and ch in DIGITS:
        return ("exp", stack, None)
    if mode in ("dot", "exp_mark", "exp_sign"):
        raise Invalid(f"expected a digit, got {ch!r}")
    if mode in NUMBER_DONE:
        mode = "after"  # the number ended before `ch`; now treat `ch` as what follows a value

    # -- between tokens --------------------------------------------------------------
    if ch in WHITESPACE:
        return (mode, stack, None)
    if mode == "after":
        if not stack:
            raise Invalid(f"{ch!r} after the end of the document")
        in_object = stack[-1] == "{"
        if ch == ",":
            return ("key" if in_object else "value", stack, None)
        if ch == ("}" if in_object else "]"):
            return ("after", stack[:-1], None)
        raise Invalid(f"expected ',' or the closing bracket, got {ch!r}")
    if mode == "colon":
        if ch == ":":
            return ("value", stack, None)
        raise Invalid(f"expected ':', got {ch!r}")
    if mode in ("key", "key_or_end"):
        if ch == '"':
            return ("string", stack, True)
        if ch == "}" and mode == "key_or_end":  # the empty object
            return ("after", stack[:-1], None)
        raise Invalid(f"expected an object key, got {ch!r}")

    # mode is "value", or "value_or_end" right after "[".
    if ch == "]" and mode == "value_or_end":  # the empty array
        return ("after", stack[:-1], None)
    if ch == '"':
        return ("string", stack, False)
    if ch == "{":
        return ("key_or_end", stack + "{", None)
    if ch == "[":
        return ("value_or_end", stack + "[", None)
    if ch == "-":
        return ("minus", stack, None)
    if ch == "0":
        return ("zero", stack, None)
    if ch in DIGITS:
        return ("int", stack, None)
    if ch in LITERALS:
        return ("literal", stack, LITERALS[ch])
    raise Invalid(f"expected a value, got {ch!r}")
