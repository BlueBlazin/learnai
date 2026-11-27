## The idea

Verifying a single chain of $K$ draft tokens takes one target forward pass because a causal transformer computes position $t$ using only tokens $0 \dots t$. Feeding the draft chain behind the prefix produces, at every draft position, the exact logits the target would have computed on its own.

A draft tree is a set of chains that share prefixes. Running each root-to-leaf path as a separate batch row would work, but it would recompute the shared branches and duplicate the prefix cache across rows. Instead, we flatten every node into one sequence and restrict attention so that each node performs the exact computation of its own root path.

In a decoder layer, position $r$ interacts with other positions only through attention, which depends on two things: which keys it attends to, and the position IDs used for positional encoding (such as RoPE rotations on queries and keys). Suppose that for every node $i$:

1. Its visible keys are the prefix, node $i$ itself, and all ancestors of $i$.
2. The prefix tokens carry positions $0, \dots, P - 1$, and every node $j$ carries position $P + d_j$, where $d_j$ is its depth.

By induction over transformer layers, the hidden state of node $i$ equals the hidden state of the final token when running ordinary causal attention on `prefix + root path of i`. The induction holds because every ancestor $j$ that node $i$ attends to also sees only the prefix and its own ancestors, which form a prefix of the same path. Every requirement in the specification (attending to all ancestors, self-attention, and depth-based position IDs) is forced by this exact equivalence.

The prefix rows remain lower-triangular and never attend to any tree node, so the prefix hidden states and KV cache are unaffected by the tree. In a serving engine, the prefix is already cached, so only the `(N, P + N)` node rows of the mask are needed, with the left `(N, P)` block set to all `True`.

**Walking the tree.** After the forward pass, `target_argmax[0]` tells us which depth-0 node (if any) the target would pick. The greedy token at that accepted node tells us which of its children is right, and so on down the tree. A node can only be accepted if its parent was accepted, because its logits were computed in a context containing its parent's token. The walk stops at the first level where no child matches the target's token (making that token a correction) or after a leaf (making it a bonus token). Either way, the round emits the accepted tokens plus one target token, matching greedy verification of a chain as the branchless special case.

## Reference solution (PyTorch)

```python
def tree_inputs(parents, prefix_len):
    n = parents.shape[0]
    node = torch.arange(n)
    anc = node
    reach = torch.zeros(n, n, dtype=torch.bool)
    steps = torch.zeros_like(parents)
    while bool((anc >= 0).any()):
        alive = anc >= 0
        reach = reach | (anc[:, None] == node[None, :])
        steps = steps + alive.to(steps.dtype)
        anc = torch.where(alive, parents[anc.clamp(min=0)], anc)
    prefix = torch.arange(prefix_len)
    causal = prefix[None, :] <= prefix[:, None]
    top = torch.cat([causal, torch.zeros(prefix_len, n, dtype=torch.bool)], dim=1)
    bottom = torch.cat([torch.ones(n, prefix_len, dtype=torch.bool), reach], dim=1)
    mask = torch.cat([top, bottom], dim=0)
    position_ids = torch.cat([prefix.to(parents.dtype), prefix_len + steps - 1])
    return mask, position_ids


def verify_tree(parents, tokens, target_argmax):
    path = []
    current = -1
    want = target_argmax[0]
    while True:
        match = (parents == current) & (tokens == want)
        if not bool(match.any()):
            return path, int(want)
        current = int(torch.argmax(match.to(torch.int64)))
        path.append(current)
        want = target_argmax[current + 1]
```

The loop climbs parent pointers across all nodes at once. After step $s$, `anc[i]` holds the ancestor $s$ levels above node $i$, or `-1` once node $i$ has left the tree. Comparing `anc` against the column indices marks one ancestor per row per step, while `-1` matches no column. Because parents always precede children, you could also fill `reach` row by row (`reach[i] = reach[parents[i]]` plus the diagonal), but the vectorized climb avoids in-place writes and translates directly to JAX and MLX.

## Common mistakes

- **Siblings visible.** Using a plain causal mask over the flat sequence lets a node attend to every earlier node, including siblings and cousins from other branches.
- **Positions by flat index.** Using `arange(P + N)` gives a node a position that depends on how many unrelated nodes precede it, scoring it at the wrong distance from the prefix under rotary or learned embeddings.
- **Parent only.** Marking only the immediate parent leaves out grandparents and higher ancestors, creating holes in the node's context.
- **No self-attention.** Every node row must attend to its own column, just like in a standard causal mask.
- **Depths starting from 1.** Depth-0 nodes sit at position $P$, immediately after prefix token $P - 1$.
- **Non-causal prefix block.** Attending to the entire prefix applies only to node rows. Prefix rows must remain lower-triangular so prefix states do not change.
- **Accepting orphans.** Checking each node independently against `target_argmax[parent + 1]` and taking the deepest match can accept nodes whose parent was rejected.
- **Trying only the first child.** Verification must check all children of the current node, since any sibling can match.
- **Misaligned `target_argmax`.** Entry `0` belongs to the prefix, so node $i$'s prediction lives at `target_argmax[i + 1]`.
- **Returning the last draft token as `next_token`.** The returned `next_token` is always the target's own prediction (either the correction or the bonus token).

## Complexity

`tree_inputs` runs $(\text{depth} + 1)$ steps of $O(N^2)$ boolean work for the tree block and uses $O((P + N)^2)$ memory for the full mask. In a serving engine, only the `(N, P + N)` node rows are materialized, or the tree block is precomputed once for a fixed tree shape. `verify_tree` does one $O(N)$ comparison per accepted level. Together, they verify $N$ candidate tokens in a single target forward pass instead of one pass per token.

## In the wild

- **Medusa** ("Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", arXiv:2401.10774) precomputes these buffers for a fixed tree of head candidates in `generate_medusa_buffers`. The mask starts as the identity, every node attends to the root and its own path, and position IDs equal node depths shifted by the current prefix length. Because Medusa's root is the base model's own next token, its single depth-0 node is always accepted. Its greedy acceptance compares each enumerated root-to-leaf path against the argmax one position earlier and keeps the longest matching prefix, which is equivalent to the walk above.
- **SpecInfer** (ASPLOS 2024, arXiv:2305.09781) introduced the token tree and one-pass verification, replacing greedy comparisons with the multi-draft rejection rule from `multi-draft-rejection`. **Sequoia** (arXiv:2402.12374) optimizes the tree shape with dynamic programming, and **EAGLE** grows it with a lightweight autoregressive head. All three rely on this exact mask and position-ID construction.
- After verification, an engine keeps only the KV-cache entries along the accepted path, gathering flat indices `P + path[j]` into consecutive slots. Skipping that compaction is the tree equivalent of failing to roll back a KV cache.
