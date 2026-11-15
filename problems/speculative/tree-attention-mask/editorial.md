## The idea

Verifying a chain of $K$ draft tokens costs one target pass because a causal
transformer computes, at position $t$, a function of tokens $0..t$ only: feeding
the drafts behind the prefix yields, at every draft position, exactly the logits
the target would have computed had it arrived there by itself.

A tree is several chains that share prefixes. Running each root-to-leaf path as
its own row of a batch would work and would recompute the shared parts (and the
whole prefix, unless its cache is copied per row). The alternative is one row that
contains every node once, with attention restricted so that each node's
computation is the computation of its own chain.

What does the output at a position depend on? In a decoder layer, position $r$
reads other positions only through attention, and only through two things: *which*
keys it may attend to, and the *positions* used by the position encoding (with
RoPE, the rotation of queries and keys). So if, for node $i$,

1. the set of visible keys is the prefix, node $i$, and its ancestors, and
2. those tokens carry the position ids $0, \dots, P - 1$ and $P + \text{depth}$,

then by induction over layers the hidden state of node $i$ equals the hidden
state of the last token of the sequence `prefix + root path of i` in an ordinary
causal run. The induction needs the same to hold for every key node $i$ reads, and
it does: an ancestor's visible set is the prefix and *its* ancestors, a subset of
the same path. This is why "ancestors" must mean all of them, why the node must
see itself, and why position ids are depths and not flat indices: none of these
is a convention, each is forced by "equal to the path run alone".

The prefix rows stay causal and never see a node, so the prefix's hidden states
(and its KV cache) are exactly those of the prefix alone. In an engine the prefix
is already in the cache; only the node rows of the mask, shape `(N, P + N)`, are
needed, and the bottom-left block is all `True`.

**Reading the result.** After the pass, the greedy token at the last prefix
position says which depth-0 node (if any) is right; the greedy token at that node
says which of its children is right; and so on. A node is accepted only if its
parent is, because its logits were computed in a context containing the parent's
token. The walk ends at the first level where no child carries the target's token
(the target's token is then the correction) or below a leaf (then it is a bonus
token). Either way the round emits the accepted tokens plus one, like greedy
verification of a chain, which is the special case of a tree without branches.

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

The climb is pointer jumping over the whole node vector: after step $s$, `anc[i]`
is the ancestor of $i$ that is $s$ levels up, or $-1$. Comparing `anc` with the
column index marks one ancestor per row per step; a value of $-1$ matches no
column. Because parents precede children, one could also fill `reach` row by row
(`reach[i] = reach[parents[i]]` plus the diagonal); the vectorized climb avoids
in-place writes and reads the same in JAX and MLX.

## Common mistakes

- **Siblings visible.** Using the plain causal mask over the flat sequence lets a
  node attend to every node flattened before it. Its logits then describe a
  sequence that contains its siblings and cousins.
- **Positions by flat index.** `arange(P + N)` gives a node a position that
  depends on how many unrelated nodes precede it. With learned or rotary positions
  the node is scored at the wrong distance from the prefix.
- **Parent only.** Marking the parent but not the grandparents: the node sees the
  prefix and one token of its path, with a hole in between.
- **No self-attention.** A row must include its own column, as in any causal
  mask.
- **Depths from 1.** Depth-0 nodes sit at position $P$, immediately after the
  prefix.
- **A non-causal prefix block.** "Everything sees the prefix" is right for node
  rows only; prefix rows remain lower-triangular, otherwise the prefix itself is
  recomputed in a context it never had.
- **Accepting orphans.** Checking each node against the target's token after its
  parent, independently, and taking the deepest match accepts nodes below a
  rejected parent.
- **Trying only the first child.** The point of the tree is that any sibling may
  be the right one.
- **Misaligned argmax.** `target_argmax[i + 1]` belongs to node $i$; entry 0 is
  the prefix's.
- **Returning the last draft token as the next token.** The next token is always
  the target's own: the correction or the bonus.

## Complexity

`tree_inputs` takes (depth + 1) steps of $O(N^2)$ boolean work for the tree block
and $O((P + N)^2)$ memory for the full mask; in an engine only the `(N, P + N)`
node rows are materialized, or the tree block is precomputed once for a fixed tree
shape. `verify_tree` takes one $O(N)$ comparison per accepted level. The forward
pass it replaces is one pass over $N$ extra tokens instead of one pass per token.

## In the wild

- **Medusa** ("Medusa: Simple LLM Inference Acceleration Framework with Multiple
  Decoding Heads", arXiv:2401.10774) precomputes these buffers for a fixed tree
  of head candidates in `generate_medusa_buffers`: the tree attention mask starts
  as the identity, every node attends to the root, and each node additionally
  attends to the nodes on its own path; position ids are the node depths, to which
  the current input length is added at decoding time. Its root is the base model's
  own next token, so its tree has one depth-0 node that is always right. Its
  greedy acceptance compares each candidate path with the argmax one position
  earlier and keeps the longest leading run, which is the walk above expressed
  over enumerated root-to-leaf paths.
- **SpecInfer** (ASPLOS 2024, arXiv:2305.09781) introduces the token tree and
  verifies it against the target in one pass; its stochastic verification
  replaces the comparison at each node by the multi-draft rejection rule of
  `multi-draft-rejection`. **Sequoia** (arXiv:2402.12374) chooses the tree shape
  by dynamic programming, and **EAGLE** grows it from a small autoregressive head;
  all three need exactly this mask.
- After acceptance, an engine keeps the cache entries of the accepted path only,
  gathered from flat indices `P + path[j]` into consecutive positions. Forgetting
  that compaction is the tree version of not rolling back a cache.
