def build_trie(choices: list[list[int]], eos_id: int):
    # A node is a dict from token id to child node. The end of a choice is stored as
    # one more edge, labeled EOS: "this choice is complete" and "EOS may come next"
    # are the same fact. A node can have both an EOS edge and ordinary children, which
    # is what happens when one choice is a prefix of another.
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
        if node is None:  # fell off the trie: nothing can follow
            return []
    return sorted(node)
