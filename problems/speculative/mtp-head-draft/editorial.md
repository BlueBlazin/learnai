## The idea

The final hidden state $h_t$ is trained to make one prediction, the token at
$t + 1$, through the output head. It nevertheless carries a good deal of
information about the tokens after that: in fluent text, the next few tokens are
often determined once the model "knows what it is about to say". A multi-token
prediction head is a small function trained to read the token at $t + 1 + j$ out
of $h_t$.

**Which design.** This problem models Medusa. Each head is

$$
p^{(j)} = \operatorname{softmax}\big(W_2^{(j)} \, (h_t + \operatorname{SiLU}(W_1^{(j)} h_t + b_1^{(j)}))\big),
$$

a residual block followed by a head-specific unembedding. The heads are
*independent*: head $j$ predicts position $t + 1 + j$ without knowing which tokens
the other heads chose for the positions in between. That is what makes them cheap
(no sequential dependence, all heads at once) and what limits them (the guess for
$t + 3$ cannot depend on the guess for $t + 2$). The residual form is there so
that a freshly initialized head, with $W_1 = 0$, passes $h_t$ through unchanged.

The contrast is DeepSeek-V3's multi-token prediction. There the extra modules are
*chained*: module $k$ takes the previous depth's hidden states together with the
embedding of the next token, runs its own transformer block, and predicts through
the output head **shared** with the main model (the embedding is shared too), so
that the causal chain is kept at every depth. Parallel versus chained, own versus
shared unembedding: these are the two axes, and a drafting routine has to match
the design the heads were trained for.

**Why `draft[0]` is special.** The model's own next token costs nothing extra and
is not a guess: under greedy decoding it is the token that will be emitted. So a
round always yields at least one token, and the heads' tokens ride along as
speculation.

**Verification and its alignment.** Feed the $k + 1$ drafted tokens to the model
in one pass. Row $j$ of that pass is computed from the sequence plus
`draft[0..j]`; its argmax is the model's token for the position *after*
`draft[j]`. The guess for that position is `draft[j + 1]`. Hence the shift by one:
`draft[j]` is confirmed by `target_argmax[j - 1]`, and only while everything
before it was confirmed, since a row computed after a wrong token describes a
sequence that will not exist.

**Why the loop is lossless and has no bonus token.** Let $n$ tokens be accepted.
Row $n - 1$ of the verification pass was computed from verified tokens only, so
its hidden state is exactly the state the model would have at that position in
ordinary decoding. The next round drafts from it; its `draft[0]` is the argmax of
the output head on that state, which is what a "bonus" or "correction" token
would have been. By induction every emitted token is the model's greedy token.
Compare with a draft-model loop: there the correction token is emitted in the same
round; here it is the first token of the next draft. One pass per round does
double duty, verifying one draft and seeding the next.

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

With the heads' weights stacked into `(k, D, D)`, `(k, D)`, and `(k, V, D)` tensors
the loop becomes three batched operations, which is how it runs in practice.

## Common mistakes

- **Chaining the heads.** Carrying a `state` variable from one head to the next
  (`state = state + silu(w1 @ state + b1)`) is the natural way to write a stack of
  residual blocks, and it is a different architecture. Parallel heads all read
  `hidden`.
- **Head index off by one.** `draft[j]` comes from head $j - 1$; head 0 already
  looks two positions ahead, because position $t + 1$ belongs to the model's own
  head. Using head $j$ for `draft[j]` puts every guess one position early.
- **A shared unembedding.** Projecting the head state through `w_out` is right
  for DeepSeek-V3's modules and wrong here.
- **Mangling the residual block.** Dropping the skip connection, applying the
  activation to the sum, dropping the bias, or using ReLU for SiLU: each agrees
  with the right formula on some inputs (a zero branch, positive
  pre-activations) and not in general.
- **Comparing at the same index.** `draft[j] == target_argmax[j]` compares a
  guess with the model's token for the *next* position.
- **Counting matches after a mismatch**, or forgetting that `draft[0]` always
  counts.

## Complexity

`mtp_draft` costs $k$ products with $D \times D$ and $V \times D$ matrices,
$O(k D (D + V))$, against one forward pass of a draft model per drafted token.
`mtp_accept` is $O(k)$. A round is one forward pass of the model over $k + 1$
tokens and emits between 1 and $k + 1$ of them.

## In the wild

- **Medusa** ("Medusa: Simple LLM Inference Acceleration Framework with Multiple
  Decoding Heads", arXiv:2401.10774). In the reference implementation a head is a
  residual block, `x + SiLU(Linear(x))` with the linear layer's weight initialized
  to zero, followed by a bias-free linear layer from the hidden size to the
  vocabulary; all heads are applied to the same last hidden state that the
  original output head reads. Medusa does not stop at one guess per head: it
  takes the top few tokens of every head, arranges them in a tree, verifies the
  tree in one pass with a tree attention mask (`tree-attention-mask`), and may
  accept by a relaxed rule (`typical-acceptance`). The top-1 chain of this
  problem is the simplest tree.
- **DeepSeek-V3** ("DeepSeek-V3 Technical Report", arXiv:2412.19437, section on
  multi-token prediction) predicts the additional tokens sequentially and keeps
  the complete causal chain at each depth; each module consists of the shared
  embedding layer, the shared output head, a transformer block, and a projection
  matrix. The report presents the modules as a training objective that can be
  dropped at inference or reused for speculative decoding, and vLLM exposes that
  use as its `mtp` speculative method.
- The loss of independence is measurable: parallel heads get less accurate with
  every position, which is why Medusa uses a handful of heads and compensates
  with several candidates per head.
