# BUG: a choice is considered complete only at a leaf of the trie; a choice that is a prefix of a longer one never gets EOS and cannot be selected.
def build_trie(choices: list[list[int]], eos_id: int):
    root: dict = {}
    for choice in choices:
        node = root
        for token in choice:
            node = node.setdefault(token, {})
    return root, eos_id


def allowed_next(trie, generated: list[int]) -> list[int]:
    node, eos_id = trie
    for token in generated:
        node = node.get(token)
        if node is None:
            return []
    if not node and generated:
        return [eos_id]
    return sorted(node)
