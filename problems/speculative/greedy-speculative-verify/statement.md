Implement the verification step of greedy speculative decoding. A fast draft model
proposes the next `K` tokens, and the target model scores all of them in one forward
pass. Your function determines how many draft tokens to accept and returns the next
verified token.

![This problem is Verify, here an argmax compared with each draft token.](assets/map.svg)

## Specification

```python
def verify_greedy(draft_tokens, target_logits): ...
```

The target model runs once on the already-verified prefix followed by the `K` tokens
in `draft_tokens`. The input `target_logits` contains the `K + 1` rows of next-token
logits over a vocabulary of size `V`:

- For $0 \le i < K$, row `i` predicts the token at position `i` of the draft. It is
  computed from the verified prefix and `draft_tokens[0..i-1]`, so row 0 depends on
  no draft token.
- Row `K` predicts the token immediately after the last draft token.

Let $g_i = \operatorname{argmax}_v\, \texttt{target\_logits}[i, v]$ be the target's
greedy token for row $i$. Accept the longest prefix of `draft_tokens` where every
token matches the target's greedy token. At the first mismatch, the target's token
for that position replaces the draft token, and all later draft tokens are discarded.
Even if a later position happens to match, its logits were computed from a context
containing the rejected token, so it cannot be kept. If all `K` draft tokens match,
row `K` supplies one extra token for free (the *bonus token*).

Formally, the accepted count `n_accepted` and the following token `next_token` are

$$
\texttt{n\_accepted} = \min\big(\{\, i : 0 \le i < K,\; g_i \ne \texttt{draft\_tokens}[i] \,\} \cup \{K\}\big),
\qquad
\texttt{next\_token} = g_{\texttt{n\_accepted}} .
$$

- `draft_tokens`: the `K` token indices proposed by the draft model.
- `target_logits`: the target model's logits for the `K + 1` positions. Entries may
  be `-inf` for masked tokens, and every row has at least one finite entry.
- Returns `(n_accepted, next_token)` as two Python `int`s, where
  $0 \le \texttt{n\_accepted} \le K$ is the number of accepted draft tokens and
  `next_token` is the token that follows them.

Each verification step extends the verified sequence by `n_accepted + 1` tokens,
consisting of `draft_tokens[:n_accepted]` followed by `next_token`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `draft_tokens` | `(K,)` | integer |
| `target_logits` | `(K + 1, V)` | float32 |
| result | `(n_accepted, next_token)` | Python `int`, Python `int` |

## Example

```python
target_logits = [[ 0.1, -1.0,  2.5,  0.3,  0.0],    # argmax 2
                 [ 1.8,  0.2, -0.5,  1.1,  0.4],    # argmax 0
                 [ 0.0,  0.5, -1.2,  2.0,  1.5],    # argmax 3
                 [-0.3,  2.2,  0.1,  0.0,  1.0]]    # argmax 1

verify_greedy([2, 0, 4], target_logits)   # (2, 3)
verify_greedy([2, 0, 3], target_logits)   # (3, 1)
verify_greedy([1, 0, 3], target_logits)   # (0, 2)
```

![The three calls of the example, draft tokens above the target's greedy tokens $g_i$. Dashed cells are discarded, and the heavy cell is `next_token`.](assets/verify-calls.svg)

- In the first call, the target agrees with the first two draft tokens. At position 2,
  the draft proposes 4 while the target picks 3, so 2 tokens are accepted and token 3
  is returned as `next_token`. Row 3 is unused.
- In the second call, all three draft tokens match. All 3 are accepted, and row 3
  provides the bonus token 1, yielding four verified tokens from one target pass.
- In the third call, the very first draft token is wrong, so 0 tokens are accepted
  and token 2 is returned. Even though positions 1 and 2 match the target's argmax,
  rows 1 and 2 were conditioned on the rejected token 1 and are invalid for the true
  sequence starting with token 2.

## Constraints

- $1 \le K \le 16$ and $2 \le V \le 64$.
- Outputs are compared for exact equality. No row has tied largest logits, and the
  largest and second-largest logits in every row differ by at least `1e-3`.
- A handful of tensor operations suffice.

## Why this matters

Greedy speculative decoding is usually the first speculative method an inference
engine ships. A small model, an n-gram lookup over the prompt, or an early-exit head
proposes several tokens, and one batched forward pass of the target model checks them
all.

The output must be identical to running the target model greedily on its own, and
this verification function enforces that guarantee. Accepting a match after an
earlier mismatch silently diverges from the target model. Misaligning logits and
drafts by one row causes almost every draft to be rejected, making decoding slower
than running without speculation. Dropping the bonus token wastes a target forward
pass on a token that was already computed.
