import jax
import jax.numpy as jnp


def tree_inputs(parents: jax.Array, prefix_len: int) -> tuple[jax.Array, jax.Array]:
    n = parents.shape[0]
    node = jnp.arange(n)
    # Climb from every node toward the top of the tree, all nodes at once. `anc` is
    # the ancestor each node has reached (itself at first), -1 once it has left the
    # tree. Every ancestor visited is a column the node may attend to, and the number
    # of steps before leaving is the node's depth plus one.
    anc = node
    reach = jnp.zeros((n, n), dtype=bool)  # reach[i, j]: j is i or an ancestor of i
    steps = jnp.zeros_like(parents)
    while bool(jnp.any(anc >= 0)):
        alive = anc >= 0
        reach = reach | (anc[:, None] == node[None, :])  # -1 matches no column
        steps = steps + alive.astype(steps.dtype)
        anc = jnp.where(alive, parents[jnp.maximum(anc, 0)], anc)
    prefix = jnp.arange(prefix_len)
    causal = prefix[None, :] <= prefix[:, None]
    # Rows: prefix, then nodes. Columns: prefix, then nodes. The prefix never sees the
    # tree; every node sees the whole prefix.
    top = jnp.concatenate([causal, jnp.zeros((prefix_len, n), dtype=bool)], axis=1)
    bottom = jnp.concatenate([jnp.ones((n, prefix_len), dtype=bool), reach], axis=1)
    mask = jnp.concatenate([top, bottom], axis=0)
    # A node sits where it would sit if its path were the only continuation.
    position_ids = jnp.concatenate([prefix.astype(parents.dtype), prefix_len + steps - 1])
    return mask, position_ids


def verify_tree(
    parents: jax.Array, tokens: jax.Array, target_argmax: jax.Array
) -> tuple[list[int], int]:
    path: list[int] = []
    current = -1  # the node the walk stands on; -1 is the end of the prefix
    want = target_argmax[0]  # the target's own token after `current`
    while True:
        # The only node that can be accepted next is a child of `current` that
        # carries exactly the token the target would have produced there.
        match = (parents == current) & (tokens == want)
        if not bool(jnp.any(match)):
            return path, int(want)
        current = int(jnp.argmax(match))
        path.append(current)
        want = target_argmax[current + 1]
