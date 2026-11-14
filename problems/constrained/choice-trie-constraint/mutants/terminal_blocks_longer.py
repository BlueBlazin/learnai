# BUG: as soon as the output equals a choice only EOS is allowed, so a longer choice that starts with a shorter one can never be completed.
END = "end"


def build_trie(choices: list[list[int]], eos_id: int):
    root: dict = {}
    for choice in choices:
        node = root
        for token in choice:
            node = node.setdefault(token, {})
        node[END] = True
    return root, eos_id


def allowed_next(trie, generated: list[int]) -> list[int]:
    node, eos_id = trie
    for token in generated:
        node = node.get(token)
        if node is None:
            return []
    if END in node:
        return [eos_id]
    return sorted(node)
