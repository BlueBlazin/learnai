from learnai.kit import DEAD, CharDFA, ToyVocab


def _walk_token(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int:
    """The state after reading the token's bytes from `state`, or DEAD."""
    piece = vocab.bytes(token)
    if not piece:
        # A special token stands for no text. Walking zero bytes would "survive" in
        # every state, so it has to be ruled out explicitly.
        return DEAD
    # Byte by byte, so a token that ends in the middle of a UTF-8 character simply
    # leaves the DFA in an intermediate state. walk() stops at the first dead byte.
    return dfa.walk(state, piece)


def allowed_tokens(dfa: CharDFA, state: int, vocab: ToyVocab) -> list[int]:
    # Surviving is the whole test: the end state need not be accepting, because later
    # tokens can still complete the string.
    return [t for t in range(vocab.size) if _walk_token(dfa, state, t, vocab) != DEAD]


def advance(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int:
    after = _walk_token(dfa, state, token, vocab)
    if after == DEAD:
        raise ValueError(f"token {token} is not allowed in state {state}")
    return after
