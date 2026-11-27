Implement `mtp_draft` and `mtp_accept` for speculative decoding with Medusa-style
multi-token prediction heads. Instead of running a separate draft model, small
extra heads read the base model's last hidden state in parallel to propose several
future tokens at once, and the model verifies all of them in a single forward pass.

![This problem is Draft, here extra heads on the model's own last hidden state.](assets/map.svg)

## Specification

```python
def mtp_draft(hidden, heads, w_out, k): ...  # -> integer tensor of shape (k + 1,)
def mtp_accept(draft, target_argmax): ...  # -> int
```

### `mtp_draft`

Suppose the base model has processed a sequence ending at position $t$ and produced
its final post-norm hidden state $h \in \mathbb{R}^D$, passed as `hidden` (where $D$
is the hidden size). Using the model's output head `w_out` ($W_\text{out} \in \mathbb{R}^{V \times D}$,
for vocabulary size $V$) and the first `k` entries of `heads` (a list of $H \ge k$
tuples `(w1, b1, w2)`), produce a draft of $k + 1$ token IDs for positions $t + 1$
through $t + 1 + k$:

- **`draft[0]`** is the base model's own greedy next token for position $t + 1$:

$$
\texttt{draft}[0] = \operatorname{argmax}(W_\text{out}\, h).
$$

- **`draft[j]`** for $1 \le j \le k$ predicts position $t + 1 + j$ using
  `heads[j - 1]`, a tuple of weights $(W_1, b_1, W_2)$. Each head applies one
  residual block with a SiLU activation to $h$ and projects the result with its
  **own** unembedding matrix $W_2$:

$$
s_j = h + \operatorname{SiLU}(W_1 h + b_1), \qquad
\texttt{draft}[j] = \operatorname{argmax}(W_2\, s_j),
\qquad \operatorname{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}.
$$

Every head reads the same vector $h$ in parallel. No head sees another head's
output state or any drafted token, so head $j - 1$ predicts position $t + 1 + j$
directly from $h$. Only the first `k` heads in `heads` are used ($1 \le k \le H$).

### `mtp_accept`

Next, the base model runs a single forward pass over the $k + 1$ drafted tokens
appended to the sequence. Entry `target_argmax[j]` is the model's greedy next token
after seeing the sequence followed by `draft[0..j]`. Thus `target_argmax[j - 1]` is
the model's prediction for the position that `draft[j]` guessed.

- `draft[0]` is always accepted because it is already the model's greedy token.
- For $1 \le j \le k$, `draft[j]` is accepted if `draft[j - 1]` was accepted and
  `draft[j] == target_argmax[j - 1]`.

Return the number of accepted tokens $n$ as a Python `int` with
$1 \le n \le k + 1$:

$$
n = \min\big(\{\, j : 1 \le j \le k,\; \texttt{draft}[j] \ne \texttt{target\_argmax}[j-1] \,\} \cup \{k + 1\}\big).
$$

The round emits `draft[:n]`. There is no separate correction or bonus token because
the verification pass already computed the hidden state at the last accepted
position (row $n - 1$). The next round drafts from that hidden state, so its
`draft[0]` will be `target_argmax[n - 1]`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `hidden` | `(D,)` | float32 |
| `w1`, `b1`, `w2` of each head | `(D, D)`, `(D,)`, `(V, D)` | float32 |
| `w_out` | `(V, D)` | float32 |
| result of `mtp_draft` | `(k + 1,)` | integer |
| `draft`, `target_argmax` | `(k + 1,)` | integer |
| result of `mtp_accept` | | Python `int` |

## Example

```python
hidden = [1.0, -1.0]
w_out = [[1, 0], [0, 1], [-1, -1]]  # V = 3, D = 2
heads = [
    ([[0, 0], [0, 0]], [0, 0], [[0, 1], [1, 1], [-1, 0]]),  # head 0
    ([[0, 0], [2, 0]], [0, 1], [[1, 0], [0, 1], [1, -1]]),  # head 1
]
mtp_draft(hidden, heads, w_out, 2)  # [0, 1, 1]
```

- **`draft[0]`:** $W_\text{out} h = [1, -1, 0]$, whose argmax is `0`.
- **`draft[1]` (head 0):** $W_1 h + b_1 = [0, 0]$ and $\operatorname{SiLU}(0) = 0$,
  so $s_1 = h = [1, -1]$ and $W_2 s_1 = [-1, 0, -1]$, whose argmax is `1`. (A
  freshly initialized head has a zero residual branch that passes $h$ through
  unchanged.)
- **`draft[2]` (head 1):** $W_1 h + b_1 = [0, 2 + 1] = [0, 3]$,
  $\operatorname{SiLU}([0, 3]) \approx [0, 2.858]$, and $s_2 \approx [1, 1.858]$.
  Then $W_2 s_2 \approx [1, 1.858, -0.858]$, whose argmax is `1`. (In this example
  $s_1 = h$, so reading $s_1$ instead of $h$ happens to give the same answer, but
  in general it does not.)

```python
mtp_accept([0, 1, 1], [1, 2, 0])  # 2
mtp_accept([0, 1, 1], [1, 1, 0])  # 3
mtp_accept([0, 1, 1], [2, 1, 0])  # 1
```

- **First call:** After `draft[0] = 0`, the model predicts `1`, which matches
  `draft[1]`. After `draft[1]`, the model predicts `2`, which disagrees with
  `draft[2] = 1`. Two tokens are accepted, and the next round starts from the
  hidden state after `draft[1]`, whose greedy next token is `2`.
- **Second call:** Both drafted guesses match, so all 3 tokens are accepted.
- **Third call:** `draft[1]` does not match `target_argmax[0] = 2`, so only
  `draft[0]` is accepted. Even though `draft[2]` equals `target_argmax[1]`, that
  row was computed after a rejected token and cannot be used.

## Constraints

- $1 \le k \le H \le 8$, $2 \le D \le 64$, $3 \le V \le 64$.
- Tokens are compared exactly. Wherever an argmax is taken in a test, the largest
  and second-largest logits differ by at least `1e-3`, so float32 rounding and
  operation order cannot change the selected token.
- **End-to-end test.** The test suite also runs a generation loop on the kit's
  [`LayeredLM`](../../../python/src/learnai/kit/layeredlm.py) using your two
  functions with both random heads and heads fitted to the model's continuation.
  The generated tokens must match the model's greedy decode, and the number of
  model passes must match what those heads allow.
- A handful of tensor operations per head is expected, and a Python loop over the
  `k` heads is fine.

## Why this matters

A separate draft model must be chosen, loaded into memory alongside the target, and
executed sequentially for every drafted token. Multi-token prediction heads avoid
all of that overhead by reusing the base model's hidden state with a few small
matrix products. Each forward pass then verifies the previous draft and produces
the hidden state for the next one.

Medusa attaches parallel heads to a frozen model, whereas DeepSeek-V3 trains
chained multi-token prediction modules jointly with the model. Those two designs
differ on the exact details tested here, and code written for one fails silently on
the other. Chaining heads that were trained in parallel, or projecting through the
shared output head instead of each head's own unembedding, turns the drafts into
noise. Verification still rejects the bad drafts, so outputs remain correct, and
the regression goes unnoticed until someone checks the acceptance rate.
