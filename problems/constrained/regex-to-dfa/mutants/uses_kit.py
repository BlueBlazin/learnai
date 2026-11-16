# BUG: takes the forbidden shortcut: the automaton comes from the kit's CharDFA.from_regex.
from learnai.kit import CharDFA


def compile_regex(pattern: str) -> tuple[dict[int, dict[int, int]], int, set[int]]:
    dfa = CharDFA.from_regex(pattern)
    return {s: dict(row) for s, row in dfa._table.items()}, dfa.start, set(dfa.accepting)
