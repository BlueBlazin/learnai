# BUG: the ids are returned in the order the trie's dict happens to hold them (insertion order, EOS wherever it was added) instead of increasing order.
def build_trie(choices: list[list[int]], eos_id: int):
    root: dict = {}
    for choice in choices:
        node = root
        for token in choice:
            node = node.setdefault(token, {})
        node.setdefault(eos_id, {})
    return root


def allowed_next(trie, generated: list[int]) -> list[int]:
    node = trie
    for token in generated:
        node = node.get(token)
        if node is None:
            return []
    return list(node)
