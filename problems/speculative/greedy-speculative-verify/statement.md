Implement the verification step of greedy speculative decoding: a cheap draft model
has guessed the next `K` tokens, the target model has scored all of them in one
forward pass, and you decide how many guesses to keep and which token comes next. The
whole step is an argmax and a comparison; the substance is the alignment of logits
with drafts, stopping at the **first** disagreement, and the extra token the target
always contributes.

![This problem is Verify, here an argmax compared with each draft token.](assets/map.svg)

## Specification

```python
def verify_greedy(draft_tokens, target_logits): ...
```

The target model was run once on the verified sequence followed by the `K` draft
tokens. `target_logits` holds the `K + 1` rows of its output that matter:

- row `i`, for $0 \le i < K$, scores **the position of `draft_tokens[i]`**: it is the
  target's next-token logits given the verified sequence and
  `draft_tokens[0..i-1]`. Row 0 depends on no draft token.
- row `K` scores the position after the last draft token.

Let $g_i = \operatorname{argmax}_v\, \texttt{target\_logits}[i, v]$ be the target's
greedy token for row $i$. Then

$$
n = \min\big(\{\, i : 0 \le i < K,\; g_i \ne \texttt{draft\_tokens}[i] \,\} \cup \{K\}\big),
\qquad
\texttt{next\_token} = g_n .
$$

In words: accept the longest prefix of the draft on which the target's greedy token
equals the draft token. At the first mismatch, the target's own token for that
position replaces the draft token, and everything after it is discarded, **including
later positions that happen to match**: their logits were computed from a context
that contains the rejected token. If all `K` draft tokens are accepted, the last row
supplies one more token for free (the *bonus token*).

- `draft_tokens`: integer tensor of shape `(K,)`, the draft model's tokens.
- `target_logits`: float32 tensor of shape `(K + 1, V)`. Entries may be `-inf`
  (masked tokens); every row has at least one finite entry.
- Returns `(n_accepted, next_token)` as two Python `int`s: the number of accepted
  draft tokens, $0 \le n \le K$, and the token that follows them.

The verified sequence therefore always grows by `n_accepted + 1` tokens:
`draft_tokens[:n_accepted]` followed by `next_token`.

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

In the first call the target agrees with the first two draft tokens. At position 2
the draft says 4 and the target says 3: two tokens are accepted and the target's
token 3 comes next. The last row is not used.

In the second call every draft token matches, so all three are accepted and the last
row gives the bonus token 1: four tokens from one target pass.

In the third call the very first draft token is wrong. Nothing is accepted, although
the draft tokens at positions 1 and 2 equal the target's argmax there: rows 1 and 2
were computed after the rejected token 1 and say nothing about the sequence that
now starts with token 2.

## Constraints

- $1 \le K \le 16$ and $2 \le V \le 64$.
- Results are compared exactly. No row has two equal largest logits; the largest and
  second-largest logit of every row differ by at least `1e-3`.
- A handful of tensor operations suffice.

## Why this matters

Greedy speculative decoding is the form of speculative decoding most engines ship
first: a small model, an n-gram lookup in the prompt, or an early-exit head proposes
several tokens, and one batched forward pass of the large model checks them all. The
output is guaranteed to be identical to the large model's own greedy decode, and that
guarantee is exactly this function. Accept a match that comes after a mismatch, and
the output silently diverges from the target model. Misalign logits and drafts by one
row, and nearly every draft is rejected: the system stays correct but runs slower than
no speculation at all. Drop the bonus token, and each round pays for a target pass
that produced a token nobody used.
