def compile_regex(pattern: str) -> tuple[dict[int, dict[int, int]], int, set[int]]:
    """Compile `pattern` to the minimal DFA over bytes that matches exactly its language.

    pattern: a regular expression in the subset of the statement; the whole input must
        match. Raises ValueError for the invalid patterns the statement lists.
    Returns (transitions, start, accepting):
        transitions: {state: {byte: next_state}} with an entry (possibly empty) for
            every state; states are ints, bytes are ints in [0, 256). A missing byte
            means the input is rejected.
        start: the start state.
        accepting: the set of accepting states.
    The automaton has no unreachable state, no state from which acceptance is
    unreachable, and no two equivalent states.
    """
    raise NotImplementedError
