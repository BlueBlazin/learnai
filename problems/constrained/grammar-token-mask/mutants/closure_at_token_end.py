# BUG: inside a token, characters are scanned without predicting and completing in between; only the item sets at token boundaries are closed, so a token whose characters belong to more than one rule (`, ` across a separator and optional whitespace) is rejected.
from whetstone.kit import Chars, Grammar, Ref, ToyVocab


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
    """Token masks for a context-free grammar, by Earley recognition.

    A state is `(columns, pending)`. `columns[i]` is the Earley item set after `i`
    characters: items `(rule, alternative, dot, origin)`, "this alternative has
    matched its first `dot` symbols, starting at character `origin`". `pending` holds
    the bytes of an unfinished character.
    """

    def __init__(self, grammar: Grammar, vocab: ToyVocab) -> None:
        self.rules = grammar.rules
        self.nullable = grammar.nullable
        self.pieces = [vocab.bytes(token) for token in range(vocab.size)]

    def start(self):
        first = {("root", alt, 0, 0) for alt in range(len(self.rules["root"]))}
        return (self._close(first, ()),), b""

    def allowed(self, state) -> list[int]:
        return [t for t, piece in enumerate(self.pieces) if piece and self._token(state, piece)]

    def advance(self, state, token: int):
        piece = self.pieces[token]
        after = self._token(state, piece) if piece else None
        if after is None:
            raise ValueError(f"token {token} is not allowed in this state")
        return after

    def is_accepting(self, state) -> bool:
        columns, pending = state
        return not pending and any(
            name == "root" and origin == 0 and dot == len(self.rules[name][alt])
            for name, alt, dot, origin in columns[-1]
        )

    # -- internals ---------------------------------------------------------------------

    def _token(self, state, piece: bytes):
        """The state after the bytes of one token, or None."""
        for i, byte in enumerate(piece):
            state = self._feed(state, byte, close=i == len(piece) - 1)
            if state is None:
                return None
        return state

    def _feed(self, state, byte: int, close: bool = True):
        """The state after one more byte, or None if no member of the language continues so."""
        columns, pending = state
        pending += bytes([byte])
        points = _code_points(pending)
        if points is None:
            return None
        low, high = points
        expected = [
            self.rules[name][alt][dot]
            for name, alt, dot, _ in columns[-1]
            if dot < len(self.rules[name][alt])
        ]
        if low < high or len(pending) < len(chr(low).encode()):
            # Inside a character: fine if some character that starts with these bytes
            # is one the grammar can read next.
            if any(isinstance(s, Chars) and s.overlaps(low, high) for s in expected):
                return columns, pending
            return None
        # A whole character: scan it, then predict and complete.
        scanned = {
            (name, alt, dot + 1, origin)
            for name, alt, dot, origin in columns[-1]
            if dot < len(self.rules[name][alt])
            and isinstance(self.rules[name][alt][dot], Chars)
            and self.rules[name][alt][dot].matches(low)
        }
        if not scanned:
            return None
        if not close:
            return (*columns, frozenset(scanned)), b""
        return (*columns, self._close(scanned, columns)), b""

    def _close(self, items, columns) -> frozenset:
        """The Earley closure of `items`, the column that follows `columns`."""
        position = len(columns)
        current = set(items)
        todo = list(items)
        while todo:
            name, alt, dot, origin = todo.pop()
            body = self.rules[name][alt]
            new = []
            if dot == len(body):
                # Completed: every item that was waiting for `name` where this one
                # began moves past it. These shared parents are what a set of
                # explicit stacks would have to copy.
                waiting = columns[origin] if origin < position else tuple(current)
                for parent in waiting:
                    p_name, p_alt, p_dot, p_origin = parent
                    p_body = self.rules[p_name][p_alt]
                    if p_dot < len(p_body) and p_body[p_dot] == Ref(name):
                        new.append((p_name, p_alt, p_dot + 1, p_origin))
            elif isinstance(body[dot], Ref):
                # Predicted: the rule may begin here. A set, so a left-recursive
                # rule predicts itself once and stops.
                target = body[dot].name
                new.extend((target, a, 0, position) for a in range(len(self.rules[target])))
                if target in self.nullable:
                    # It may match nothing. Its own completion would only advance
                    # the items already in the set; this one must not depend on
                    # having been added first.
                    new.append((name, alt, dot + 1, origin))
            for item in new:
                if item not in current:
                    current.add(item)
                    todo.append(item)
        return frozenset(current)
