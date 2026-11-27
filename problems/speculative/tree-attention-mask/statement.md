Build the attention mask and position IDs that let a target model verify a whole **tree** of draft tokens in one forward pass, then read the accepted path from the model's greedy predictions. Flattening the tree into a single sequence behind the prefix avoids recomputing shared branches. The mask and position IDs must ensure that every node produces the exact same logits it would produce if its own root-to-node path were run alone.

![This problem is the single target call, which here checks a whole tree of drafts.](assets/map.svg)

## Specification

```python
def tree_inputs(parents, prefix_len): ...  # -> (mask, position_ids)
def verify_tree(parents, tokens, target_argmax): ...  # -> (path, next_token)
```

**The draft tree.** A draft tree has $N \ge 1$ nodes numbered in topological order, sitting behind a prefix of length $P = \texttt{prefix\_len} \ge 1$. Node $i$ carries the draft token `tokens[i]` and has parent `parents[i]`:

- If `parents[i] == -1`, node $i$ continues the prefix directly (depth 0).
- Otherwise, $0 \le \texttt{parents[i]} < i$, and node $i$ continues node `parents[i]`.

The *root path* of node $i$ is the chain of nodes from depth 0 down to $i$. Its *ancestors* are all nodes on that path other than $i$, and its depth $d_i$ is the number of ancestors. The flattened model input consists of the $P$ prefix tokens followed by the $N$ node tokens in index order, so node $i$ sits at flat index $P + i$.

![The example tree behind a prefix of two tokens, and the mask of the flattened sequence. A node's row is filled for the prefix, itself, and its ancestors only.](assets/tree-mask.svg)

### `tree_inputs`

Returns the boolean attention mask and integer position IDs for the flattened sequence of length $P + N$. Entry `mask[r, c]` is `True` when the query token at flat index `r` may attend to the key token at flat index `c`.

| row | may attend to |
|---|---|
| prefix row $r < P$ | prefix columns $c \le r$ (ordinary causal attention) and no tree nodes |
| node row $P + i$ | all prefix columns, column $P + i$ itself, and column $P + j$ for every ancestor $j$ of $i$ |

For position IDs, `position_ids[r]` is $r$ for each prefix row $r < P$, and $P + d_i$ for each node $i$ at flat index $P + i$. This matches the position node $i$ would occupy if the prefix and its root path were one ordinary sequence. Siblings share the same position ID.

### `verify_tree`

Assume the target model ran once on the flattened sequence using the mask and position IDs above, and its greedy next-token predictions were collected into `target_argmax`:

- `target_argmax[0]` is the target's greedy next token after the prefix.
- `target_argmax[i + 1]` is the target's greedy next token after node $i$ (conditioned on the prefix and the root path of node $i$).

Verification walks down the tree starting from the prefix. Set `current = -1` (the end of the prefix) and `want = target_argmax[0]`. Repeat the following step:

- If some node $c$ satisfies `parents[c] == current` and `tokens[c] == want`, accept $c$. Append $c$ to `path`, set `current = c`, and set `want = target_argmax[c + 1]`.
- Otherwise, stop.

Return `(path, next_token)`, where `path` is the list of accepted node indices from depth 0 down (empty if no depth-0 node matches `target_argmax[0]`), and `next_token` is the final value of `want`. That final token is the target's correction when the walk stops at a mismatch, or a bonus token when the walk reaches a leaf.

Children of the same parent always carry distinct tokens, so at most one child can match at each step. A node can only be accepted through its parent. Even if a node's token matches the target's prediction after its parent, that node is ignored unless its parent was accepted.

Together, the round emits `[tokens[i] for i in path] + [next_token]`, which matches the target model's own greedy decoding after the prefix. In a serving engine, the KV-cache entries at flat indices `P + path[j]` are then compacted into positions `P + j`, and all other tree entries are discarded.

## Shapes

| name | shape | dtype |
|---|---|---|
| `parents`, `tokens` | `(N,)` | integer |
| `prefix_len` | scalar | Python `int` ($P \ge 1$) |
| `mask` | `(P + N, P + N)` | bool |
| `position_ids` | `(P + N,)` | integer |
| `target_argmax` | `(N + 1,)` | integer |
| result of `verify_tree` | `(path, next_token)` | list of Python `int`, Python `int` |

## Example

Using the tree from the second figure with $P = 2$:

```python
parents = [-1, -1, 0, 0, 1, 2]
tokens = [7, 3, 4, 9, 8, 6]

mask, position_ids = tree_inputs(parents, 2)
# columns:  p0 p1  0  1  2  3  4  5
# p0         1  .  .  .  .  .  .  .
# p1         1  1  .  .  .  .  .  .
# node 0     1  1  1  .  .  .  .  .
# node 1     1  1  .  1  .  .  .  .
# node 2     1  1  1  .  1  .  .  .      parent 0
# node 3     1  1  1  .  .  1  .  .      parent 0
# node 4     1  1  .  1  .  .  1  .      parent 1
# node 5     1  1  1  .  1  .  .  1      parent 2, grandparent 0
# position_ids = [0, 1, 2, 2, 3, 3, 3, 4]
```

Node 3 cannot attend to its sibling node 2 even though node 2 appears earlier in the flattened sequence, and node 4 cannot attend to node 0. Nodes 2, 3, and 4 are all the second token after the prefix, so all three have position ID 3.

```python
# target_argmax: the target's token after the prefix, then after node 0, 1, ..., 5
verify_tree(parents, tokens, [7, 9, 8, 6, 2, 5, 1])  # ([0, 3], 2)
verify_tree(parents, tokens, [7, 4, 8, 6, 2, 5, 1])  # ([0, 2, 5], 1)
verify_tree(parents, tokens, [5, 9, 8, 6, 2, 5, 1])  # ([], 5)
```

- **First call:** After the prefix, the target predicts `7`, which matches depth-0 node 0. After node 0, the target predicts `9`, which matches its child node 3. Node 3 is a leaf, so the walk ends and the target's token after node 3 (`2`) becomes the bonus token. The round emits `[7, 9, 2]`. Node 4 also matches the target's prediction after node 1 (`8`), but node 1 was never accepted, so node 4 is ignored.
- **Second call:** After node 0, the target predicts `4` (node 2). After node 2, it predicts `6` (node 5). Node 5 is a leaf, so the round emits `[7, 4, 6, 1]`.
- **Third call:** The target's first token is `5`, which matches no depth-0 node. Nothing is accepted, and the round emits `[5]`.

## Constraints

- $1 \le N \le 64$ and $1 \le P \le 64$, and `parents[i] < i` always holds.
- `tree_inputs` is checked entry by entry against the specification.
- **Equivalence check:** A small decoder with rotary position embeddings (the kit's [`decoder_block_ref`](../../../python/src/learnai/kit/decoder.py)) runs once over the prefix and flattened tree with your mask and position IDs. The logits at every node must match, within `1e-4`, the logits at the last position of an ordinary causal run over the prefix followed by that node's root path. The prefix logits must remain unchanged by the tree.
- `verify_tree` is checked for exact equality. Sibling tokens are distinct in every test.
- You do not need a Python loop over pairs of nodes to build the mask. Climbing the parent pointers of all nodes at once takes as many steps as the depth of the tree.

## Why this matters

A single draft chain stops at its first wrong token. A draft tree hedges by proposing several candidates per position so that at least one branch usually survives multiple levels. That only pays off when the target model can verify all branches in a single forward pass, which is what the tree attention mask enables. Medusa, SpecInfer, Sequoia, and EAGLE all build these buffers to verify token trees.

Bugs in tree verification fail silently. If a node attends to a sibling or uses flat indices as position IDs, the target scores that node in a context that never existed. Its predictions still look plausible and tokens still get accepted, but the output no longer matches the target model. Accepting a node whose parent was rejected causes the same silent corruption even when the mask is completely right.
