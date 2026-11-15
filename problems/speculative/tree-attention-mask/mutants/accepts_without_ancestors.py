# BUG: accepts every node whose token equals the target's greedy token after its parent, without requiring the parent to be accepted, and returns the root path of the deepest such node.
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
    # Each node on its own: does it carry the target's token for its position?
    ok = tokens == target_argmax[parents + 1]
    if not bool(ok.any()):
        return [], int(target_argmax[0])
    # Depth of every node, then the deepest matching one (the lowest index on ties).
    depth = torch.zeros_like(parents)
    for i in range(parents.shape[0]):
        if int(parents[i]) >= 0:
            depth[i] = depth[parents[i]] + 1
    best = int(torch.argmax(torch.where(ok, depth + 1, torch.zeros_like(depth))))
    path = [best]
    while int(parents[path[0]]) >= 0:
        path.insert(0, int(parents[path[0]]))
    return path, int(target_argmax[best + 1])
