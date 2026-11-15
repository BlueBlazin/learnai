# BUG: reads the target's token for node i from target_argmax[i] instead of target_argmax[i + 1]: every lookup after the first is shifted by one node.
import torch


def tree_inputs(parents: torch.Tensor, prefix_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = parents.shape[0]
    node = torch.arange(n)
    # Climb from every node toward the top of the tree, all nodes at once. `anc` is
    # the ancestor each node has reached (itself at first), -1 once it has left the
    # tree. Every ancestor visited is a column the node may attend to, and the number
    # of steps before leaving is the node's depth plus one.
    anc = node
    reach = torch.zeros(n, n, dtype=torch.bool)  # reach[i, j]: j is i or an ancestor of i
    steps = torch.zeros_like(parents)
    while bool((anc >= 0).any()):
        alive = anc >= 0
        reach = reach | (anc[:, None] == node[None, :])  # -1 matches no column
        steps = steps + alive.to(steps.dtype)
        anc = torch.where(alive, parents[anc.clamp(min=0)], anc)
    prefix = torch.arange(prefix_len)
    causal = prefix[None, :] <= prefix[:, None]
    # Rows: prefix, then nodes. Columns: prefix, then nodes. The prefix never sees the
    # tree; every node sees the whole prefix.
    top = torch.cat([causal, torch.zeros(prefix_len, n, dtype=torch.bool)], dim=1)
    bottom = torch.cat([torch.ones(n, prefix_len, dtype=torch.bool), reach], dim=1)
    mask = torch.cat([top, bottom], dim=0)
    # A node sits where it would sit if its path were the only continuation.
    position_ids = torch.cat([prefix.to(parents.dtype), prefix_len + steps - 1])
    return mask, position_ids


def verify_tree(
    parents: torch.Tensor, tokens: torch.Tensor, target_argmax: torch.Tensor
) -> tuple[list[int], int]:
    path: list[int] = []
    current = -1  # the node the walk stands on; -1 is the end of the prefix
    want = target_argmax[0]  # the target's own token after `current`
    while True:
        # The only node that can be accepted next is a child of `current` that
        # carries exactly the token the target would have produced there.
        match = (parents == current) & (tokens == want)
        if not bool(match.any()):
            return path, int(want)
        current = int(torch.argmax(match.to(torch.int64)))
        path.append(current)
        want = target_argmax[current]
