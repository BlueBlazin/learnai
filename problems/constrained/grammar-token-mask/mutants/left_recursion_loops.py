# BUG: a pushdown automaton that expands the rule at the top of each stack recursively; on a left-recursive rule the expansion puts the same rule on top again and never reaches a character, so it recurses until Python gives up.
from learnai.kit import Chars, Grammar, Ref, ToyVocab


def _code_points(data: bytes) -> tuple[int, int] | None:
    """The code points whose UTF-8 encoding starts with `data` (1 to 4 bytes), as an
    inclusive range, or None if no character's encoding starts that way. For a
    complete encoding the range is that one code point."""
    lead = data[0]
    if lead < 0x80:
        length, value = 1, lead
    elif 0xC2 <= lead <= 0xDF:
        length, value = 2, lead & 0x1F
    elif 0xE0 <= lead <= 0xEF:
        length, value = 3, lead & 0x0F
    elif 0xF0 <= lead <= 0xF4:
        length, value = 4, lead & 0x07
    else:
        return None  # a continuation byte, or a byte no encoding starts with
    if len(data) > length or any(not 0x80 <= byte <= 0xBF for byte in data[1:]):
        return None
    for byte in data[1:]:
        value = (value << 6) | (byte & 0x3F)
    missing = 6 * (length - len(data))  # payload bits still to come
    low, high = value << missing, (value << missing) | ((1 << missing) - 1)
    # Only characters that really take `length` bytes (no overlong forms), nothing
    # above U+10FFFF, and no surrogates.
    low, high = max(low, (0, 0x80, 0x800, 0x10000)[length - 1]), min(high, 0x10FFFF)
    if 0xD800 <= low and high <= 0xDFFF:
        return None
    if low < 0xD800 <= high:
        high = 0xD7FF
    return (low, high) if low <= high else None


class GrammarMatcher:
    """Token masks by a pushdown automaton, llama.cpp style.

    A stack is a tuple of the symbols still to be matched, next symbol first. A state
    is `(stacks, pending)`: every stack the text so far can leave behind, and the bytes of an
    unfinished character.
    """

    def __init__(self, grammar: Grammar, vocab: ToyVocab) -> None:
        self.rules = grammar.rules
        self.pieces = [vocab.bytes(token) for token in range(vocab.size)]

    def start(self):
        return self._all(((Ref("root"),),)), b""

    def allowed(self, state) -> list[int]:
        return [t for t, piece in enumerate(self.pieces) if piece and self._token(state, piece)]

    def advance(self, state, token: int):
        piece = self.pieces[token]
        after = self._token(state, piece) if piece else None
        if after is None:
            raise ValueError(f"token {token} is not allowed in this state")
        return after

    def is_accepting(self, state) -> bool:
        stacks, pending = state
        return not pending and () in stacks

    def _expand(self, stack: tuple, out: list) -> None:
        """Replace rule references at the top of `stack` by their alternatives, until
        the top is a character set or the stack is empty; collect the results."""
        if not stack or isinstance(stack[0], Chars):
            if stack not in out:
                out.append(stack)
            return
        for alternative in self.rules[stack[0].name]:
            self._expand(alternative + stack[1:], out)

    def _token(self, state, piece: bytes):
        for byte in piece:
            state = self._feed(state, byte)
            if state is None:
                return None
        return state

    def _feed(self, state, byte: int):
        stacks, pending = state
        pending += bytes([byte])
        points = _code_points(pending)
        if points is None:
            return None
        low, high = points
        if low < high or len(pending) < len(chr(low).encode()):
            if any(stack and stack[0].overlaps(low, high) for stack in stacks):
                return stacks, pending
            return None
        after = self._all(s[1:] for s in stacks if s and s[0].matches(low))
        return (after, b"") if after else None

    def _all(self, stacks) -> tuple:
        out: list = []
        for stack in stacks:
            self._expand(stack, out)
        return tuple(out)
