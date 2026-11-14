def build_trie(choices: list[list[int]], eos_id: int):
    """Build a prefix tree over the token sequences of the choices.

    choices: one list of token ids per choice. There is at least one choice, every
        choice has at least one token, and no choice contains `eos_id`. The same
        choice may be listed more than once, and one choice may be a prefix of another.
    eos_id: the id of the end-of-sequence token.
    Returns the trie, in any representation you like: the tests only pass it back to
    `allowed_next`.
    """
    raise NotImplementedError


def allowed_next(trie, generated: list[int]) -> list[int]:
    """The token ids that may follow `generated`.

    trie: what `build_trie` returned.
    generated: the token ids produced so far (may be empty; never contains EOS).
    Returns a list of distinct ints in increasing order: every token `t` such that
    `generated + [t]` is a prefix of some choice, plus `eos_id` exactly when
    `generated` equals a choice. If `generated` is not a prefix of any choice, the
    list is empty.
    """
    raise NotImplementedError
