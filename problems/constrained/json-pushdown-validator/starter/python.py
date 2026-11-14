class Invalid(ValueError):
    """Raised by json_step when the text can no longer become valid JSON."""


def json_start():
    """The state before any character has been read.

    A state can be any value you like, but json_step must never modify one: build
    states from immutable parts (tuples, strings, ints).
    """
    raise NotImplementedError


def json_step(state, ch: str):
    """Read one more character.

    state: a state returned by json_start or json_step.
    ch: a string of exactly one character.
    Returns the new state if the text read so far, including `ch`, is a prefix of
    some valid JSON text (RFC 8259). Raises Invalid otherwise. `state` itself is left
    unchanged and stays usable.
    """
    raise NotImplementedError


def json_complete(state) -> bool:
    """Whether the text read so far is, as it stands, a complete valid JSON text."""
    raise NotImplementedError
