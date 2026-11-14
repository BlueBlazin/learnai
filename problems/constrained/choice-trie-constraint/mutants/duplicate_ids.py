# BUG: scans the list of choices and collects the next token of every matching choice without removing duplicates, so a token shared by several choices (or a choice listed twice) appears several times.
def build_trie(choices: list[list[int]], eos_id: int):
    return [list(choice) for choice in choices], eos_id


def allowed_next(trie, generated: list[int]) -> list[int]:
    choices, eos_id = trie
    n = len(generated)
    out = []
    for choice in choices:
        if choice[:n] == generated:
            out.append(choice[n] if len(choice) > n else eos_id)
    return sorted(out)
