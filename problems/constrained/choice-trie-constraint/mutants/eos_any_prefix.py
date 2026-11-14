# BUG: EOS is allowed at every node of the trie, not only where a choice ends, so the output may stop on a proper prefix of a choice (or before the first token).
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
    return sorted({*node, eos_id})
