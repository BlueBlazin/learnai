# BUG: looks only at the position, not at the path: every choice longer than the output contributes its token at index len(generated), whether or not the choice starts with the tokens generated so far.
def build_trie(choices: list[list[int]], eos_id: int):
    return [list(choice) for choice in choices], eos_id


def allowed_next(trie, generated: list[int]) -> list[int]:
    choices, eos_id = trie
    n = len(generated)
    if not any(choice[:n] == generated for choice in choices):
        return []
    out = {choice[n] for choice in choices if len(choice) > n}
    if generated in choices:
        out.add(eos_id)
    return sorted(out)
