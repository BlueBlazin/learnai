## The idea

A language model's final hidden state $h_t$ at position $t$ is trained to predict
the next token at $t + 1$ through the output head. Even so, $h_t$ carries strong
signal about the next few tokens beyond $t + 1$, because in fluent text several
upcoming words are often predictable once the model commits to a phrase. A
multi-token prediction head is a lightweight module trained to read token
$t + 1 + j$ directly from $h_t$.

**Medusa's parallel design.** This problem models Medusa. Each head $j$ computes

$$
p^{(j)} = \operatorname{softmax}\big(W_2^{(j)} \, (h_t + \operatorname{SiLU}(W_1^{(j)} h_t + b_1^{(j)}))\big),
$$

which is a single residual block followed by a head-specific unembedding matrix.
The heads are *independent*: head $j$ predicts position $t + 1 + j$ without knowing
what the earlier heads predicted for the positions in between. Independence makes
drafting cheap because all heads run in parallel, but it also limits accuracy since
the guess for $t + 3$ cannot condition on the guess for $t + 2$. The residual skip
connection ensures that a newly initialized head with $W_1 = 0$ passes $h_t$
through unchanged.

By contrast, DeepSeek-V3's multi-token prediction modules are *chained*. Each
module takes the previous depth's hidden state together with the embedding of the
next token, runs its own transformer block, and projects through the main model's
**shared** output head (the embedding table is shared too). That preserves the
causal chain at every depth. Parallel versus chained states and separate versus
shared unembeddings are the two key architectural choices, and a drafting routine
must match how the heads were trained.

**Why `draft[0]` is special.** The base model's own greedy next token costs nothing
extra and is not a speculation. Under greedy decoding, `draft[0]` is guaranteed to
be emitted, so every round produces at least one token while the heads' guesses
ride along as speculation.

**Verification alignment.** Feeding all $k + 1$ drafted tokens to the base model in
one pass produces $k + 1$ next-token predictions. Row $j$ of that pass conditions
on the prefix plus `draft[0..j]`, so its argmax is the model's prediction for the
position *after* `draft[j]`. That position was guessed by `draft[j + 1]`. Shifting
the index by one, `draft[j]` is confirmed by `target_argmax[j - 1]`, provided every
earlier draft was also confirmed. Any row computed after a wrong draft conditions
on a prefix that will not be emitted and must be ignored.

**Why the loop is lossless without a bonus token.** Suppose $n$ tokens are
accepted. Row $n - 1$ of the verification pass conditioned only on verified tokens,
so its hidden state is identical to the state the model would reach during standard
greedy decoding. The next round drafts from that hidden state, making its
`draft[0]` equal to `target_argmax[n - 1]`. By induction, every emitted token
matches the base model's greedy decode. Whereas a draft-model loop emits a
correction or bonus token at the end of the current round, a head-based loop emits
that token as `draft[0]` of the next round. One forward pass per round both
verifies the previous draft and seeds the next one.

## Reference solution (PyTorch)

```python
def mtp_draft(hidden, heads, w_out, k):
    tokens = [torch.argmax(w_out @ hidden)]
    for w1, b1, w2 in heads[:k]:
        z = w1 @ hidden + b1
        state = hidden + z * torch.sigmoid(z)
        tokens.append(torch.argmax(w2 @ state))
    return torch.stack(tokens)


def mtp_accept(draft, target_argmax):
    match = (draft[1:] == target_argmax[:-1]).to(torch.int64)
    return 1 + int(torch.cumprod(match, dim=0).sum())
```

In production, the heads' weights are stacked into `(k, D, D)`, `(k, D)`, and
`(k, V, D)` tensors so the loop becomes three batched matrix operations.

## Common mistakes

- **Chaining the heads.** Updating a running `state` across iterations
  (`state = state + silu(w1 @ state + b1)`) implements a stacked residual network
  rather than parallel heads. Every head must read the original `hidden`.
- **Head index off by one.** `draft[j]` comes from `heads[j - 1]`. Head 0 predicts
  two positions ahead ($t + 2$), because position $t + 1$ belongs to `w_out`.
  Using head $j$ for `draft[j]` shifts every prediction one position early.
- **Using a shared unembedding.** Projecting each head's state through `w_out` is
  correct for DeepSeek-V3's modules, not for Medusa heads.
- **Altering the residual block.** Dropping the skip connection, applying SiLU to
  the sum, omitting `b1`, or substituting ReLU for SiLU can match the right output
  on special inputs (such as zero weights or positive pre-activations) while
  failing in general.
- **Comparing at the same index.** Checking `draft[j] == target_argmax[j]` compares
  a guess against the model's prediction for the *next* position.
- **Counting matches after a mismatch, or forgetting `draft[0]`.** Acceptance stops
  at the first mismatch, and `draft[0]` always counts toward the returned total.

## Complexity

`mtp_draft` performs $k$ matrix-vector products of size $D \times D$ and $V \times D$,
taking $O(k D (D + V))$ time compared to running a full draft model once per
drafted token. `mtp_accept` runs in $O(k)$ time. Each decoding round runs one
forward pass of the base model over $k + 1$ tokens and emits between $1$ and
$k + 1$ tokens.

## In the wild

- **Medusa** ("Medusa: Simple LLM Inference Acceleration Framework with Multiple
  Decoding Heads", arXiv:2401.10774) defines each head as a residual block
  `x + SiLU(Linear(x))` with zero-initialized linear weights, followed by a
  bias-free linear unembedding from the hidden dimension to the vocabulary. All
  heads read the same final hidden state as the base output head. In practice,
  Medusa takes the top several candidates from each head, arranges them into a
  candidate tree, verifies the tree in one pass with a tree attention mask
  (`tree-attention-mask`), and optionally uses a relaxed acceptance rule
  (`typical-acceptance`). The top-1 chain in this problem is the simplest case of
  that tree.
- **DeepSeek-V3** ("DeepSeek-V3 Technical Report", arXiv:2412.19437, section on
  multi-token prediction) predicts extra tokens sequentially to keep the full
  causal chain at each depth. Each module combines the shared embedding layer, the
  shared output head, a transformer block, and a projection matrix. These modules
  serve as a training objective that can either be dropped at inference or reused
  for speculative decoding (exposed in vLLM as the `mtp` speculative method).
  Drafting with a chained module is covered in `mtp-chain-draft`.
- Independence has a measurable cost. Parallel heads lose accuracy with each extra
  step ahead, so Medusa uses only a few heads and compensates by verifying multiple
  candidates per head in a tree.
