import mlx.core as mx


def tree_inputs(parents: mx.array, prefix_len: int) -> tuple[mx.array, mx.array]:
    n = parents.shape[0]
    node = mx.arange(n)
    # Climb from every node toward the top of the tree, all nodes at once. `anc` is
    # the ancestor each node has reached (itself at first), -1 once it has left the
    # tree. Every ancestor visited is a column the node may attend to, and the number
    # of steps before leaving is the node's depth plus one.
    anc = node
    reach = mx.zeros((n, n), dtype=mx.bool_)  # reach[i, j]: j is i or an ancestor of i
    steps = mx.zeros_like(parents)
    while mx.any(anc >= 0).item():
        alive = anc >= 0
        reach = reach | (anc[:, None] == node[None, :])  # -1 matches no column
        steps = steps + alive.astype(steps.dtype)
        anc = mx.where(alive, parents[mx.maximum(anc, 0)], anc)
    prefix = mx.arange(prefix_len)
    causal = prefix[None, :] <= prefix[:, None]
    # Rows: prefix, then nodes. Columns: prefix, then nodes. The prefix never sees the
    # tree; every node sees the whole prefix.
    top = mx.concatenate([causal, mx.zeros((prefix_len, n), dtype=mx.bool_)], axis=1)
    bottom = mx.concatenate([mx.ones((n, prefix_len), dtype=mx.bool_), reach], axis=1)
    mask = mx.concatenate([top, bottom], axis=0)
    # A node sits where it would sit if its path were the only continuation.
    position_ids = mx.concatenate([prefix.astype(parents.dtype), prefix_len + steps - 1])
    return mask, position_ids


def verify_tree(
    parents: mx.array, tokens: mx.array, target_argmax: mx.array
) -> tuple[list[int], int]:
    path: list[int] = []
    current = -1  # the node the walk stands on; -1 is the end of the prefix
    want = target_argmax[0]  # the target's own token after `current`
    while True:
        # The only node that can be accepted next is a child of `current` that
        # carries exactly the token the target would have produced there.
        match = (parents == current) & (tokens == want)
        if not mx.any(match).item():
            return path, int(want.item())
        current = int(mx.argmax(match).item())
        path.append(current)
        want = target_argmax[current + 1]
