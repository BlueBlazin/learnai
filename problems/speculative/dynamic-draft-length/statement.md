Implement the policy that decides how many tokens a draft model proposes in each
round of speculative decoding, as Hugging Face's assisted generation does it: an
additive rule that lengthens the draft after a fully accepted round and shortens it
after a rejection, plus a confidence rule that ends a draft early. Then run the
policy over recorded acceptance traces and total up the cost. No single formula is
hard; the substance is which count the growth rule compares with, what happens to
the token that triggers the confidence stop, and the end of the output.

![This problem is Adapt k, which sets the draft length of the next round.](assets/map.svg)

## Specification

```python
def should_stop_drafting(logits, threshold): ...  # -> bool
def next_k(k, n_accepted, n_drafted): ...  # -> int
def simulate(trace, k, schedule, threshold, c_draft, c_target): ...  # -> dict
```

**`should_stop_drafting(logits, threshold)`.** `logits` is the draft model's
next-token logits (a non-empty list of floats) for the token it has just drafted.
The draft model decodes greedily, so that token is the argmax, and the draft's
*confidence* in it is its softmax probability:

$$
\text{confidence} = \max_v \frac{e^{\,\ell_v}}{\sum_u e^{\,\ell_u}} .
$$

Return `True` when the confidence is **strictly below** `threshold`, else `False`.
A threshold of `0.0` therefore never stops. Logits may be as large as $\pm 10^4$.

**`next_k(k, n_accepted, n_drafted)`.** A round drafted `n_drafted >= 1` tokens and
the target accepted the first `n_accepted` of them. The draft length for the next
round is

$$
\text{next\_k} = \begin{cases}
k + 2 & \text{if } n_\text{accepted} = n_\text{drafted} \quad \text{(every drafted token was accepted)}, \\
\max(1,\ k - 1) & \text{otherwise.}
\end{cases}
$$

The comparison is with the number of tokens **actually drafted**, which can be
smaller than `k`. There is no upper limit.

**`simulate(trace, k, schedule, threshold, c_draft, c_target)`.** `trace` describes
the generation of `T = len(trace)` tokens. `trace[t]` is a pair `(match, logits)` for
output position `t`:

- `match` (a `bool`): whether the draft model's token at position `t` equals the
  target model's token there, given that all earlier positions are correct;
- `logits`: the draft model's logits at position `t`, as for `should_stop_drafting`.

The trace is a model of a decoding run. Within a round the draft continues from its
own tokens; the simulation reads the logits of a drafted position from the trace
whether or not an earlier token of the same round was wrong.

Starting at position `pos = 0` with draft length `k`, repeat until `pos == T`:

1. **Limit.** `limit = min(k, T - pos - 1)`. The target always contributes one token
   per round, so at most `T - pos - 1` drafted tokens can be used.
2. **Draft.** Draft tokens for positions `pos, pos + 1, ...` one at a time. After
   each token, stop if `limit` tokens have been drafted or if
   `should_stop_drafting(logits, threshold)` is true for that token's logits. The
   token that triggers the stop **is part of the draft**. Call the number of drafted
   tokens `n_drafted` ($0 \le n_\text{drafted} \le \text{limit}$; it is 0 only when
   `limit` is 0).
3. **Verify.** One target call. `n_accepted` is the length of the longest prefix of
   the drafted positions whose `match` is true. Matches after the first mismatch do
   not count.
4. **Account.** The round costs `n_drafted * c_draft + c_target`, and produces
   `n_accepted + 1` tokens: `pos += n_accepted + 1`.
5. **Adapt.** If `schedule == "heuristic"` and `n_drafted >= 1`, set
   `k = next_k(k, n_accepted, n_drafted)`. If `schedule == "constant"`, or if nothing
   was drafted, `k` keeps its value.

Return a dict with exactly these keys:

| key | type | value |
|---|---|---|
| `"cost"` | `int` | the sum of the costs of all rounds |
| `"rounds"` | `int` | the number of rounds (target calls) |
| `"draft_tokens"` | `int` | the sum of `n_drafted` over all rounds |
| `"ks"` | `list[int]` | the value of `k` at the start of every round, in order |

Arguments: `k` is an int $\ge 1$; `schedule` is `"constant"` or `"heuristic"`;
`threshold` is a float in $[0, 1]$; `c_draft` and `c_target` are positive ints.

## Example

```python
T, F = True, False
trace = [
    (T, [3.0, 0.0]),
    (T, [3.0, 0.0]),
    (T, [2.0, 0.0]),
    (F, [0.2, 0.0]),
    (T, [3.0, 0.0]),
    (F, [0.0, 0.0, 0.0]),
    (T, [3.0, 0.0]),
    (T, [3.0, 0.0]),
    (T, [3.0, 0.0]),
    (T, [3.0, 0.0]),
]

should_stop_drafting([3.0, 0.0], 0.4)  # False: confidence 0.953
should_stop_drafting([0.0, 0.0, 0.0], 0.4)  # True:  confidence 0.333

simulate(trace, k=2, schedule="heuristic", threshold=0.4, c_draft=1, c_target=10)
# {"cost": 59, "rounds": 5, "draft_tokens": 9, "ks": [2, 4, 3, 2, 4]}
```

| round | `pos` | `k` | limit | drafted positions | `n_drafted` | `n_accepted` | cost | next `k` |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 2 | 2 | 0, 1 | 2 | 2 | 12 | 4 |
| 2 | 3 | 4 | 4 | 3, 4, 5 (unsure: stop) | 3 | 0 | 13 | 3 |
| 3 | 4 | 3 | 3 | 4, 5 (unsure: stop) | 2 | 1 | 12 | 2 |
| 4 | 6 | 2 | 2 | 6, 7 | 2 | 2 | 12 | 4 |
| 5 | 9 | 4 | 0 | none | 0 | 0 | 10 | 4 |

Round 1 drafts two tokens, both match, and the target adds position 2: three tokens,
and `k` grows to 4. Round 2 starts at position 3. The draft is confident there
($0.55$) and wrong; it goes on to position 5, where its confidence is $1/3$, below
$0.4$: that token is the third and last of the draft. Verification rejects the
first drafted token, so nothing is accepted, the target supplies position 3, and `k`
shrinks to 3. Round 3 drafts positions 4 and 5 and stops for the same reason;
position 4 is accepted, position 5 is not. Round 5 has one token left, which the
target supplies on its own, so nothing is drafted and `k` stays 4.

Decoding the ten tokens with the target alone would cost $10 \times 10 = 100$.

## Constraints

- $1 \le T \le 400$; logit lists have 1 to 16 entries with $|\ell| \le 10^4$;
  $1 \le k \le 32$; costs are at most 1000.
- All results are compared **exactly**, including their types (`bool`, `int`, a dict
  of `int`s and a `list` of `int`s). In every test the confidence of a position
  differs from the threshold by at least $10^{-6}$, except in one test of the strict
  inequality, where the logits are equal and the confidence is exactly $1/2$ or
  $1/4$.
- **Cost.** On traces that alternate between phases where the draft is almost always
  right and phases where it is usually wrong and unsure, `simulate` with
  `schedule="heuristic"`, `threshold=0.4`, `c_draft=1`, and `c_target=10` must report
  a cost of at most $0.9$ times the cost of the best constant schedule without
  confidence stopping (the minimum over $k = 1, \dots, 32$ of
  `simulate(trace, k, "constant", 0.0, 1, 10)`, computed by the judge and rounded
  down). The specified policy costs about $0.8$ to $0.85$ of it on those traces.
- Only the standard library is needed.

## Why this matters

The draft length is the one knob of speculative decoding that has no good constant
value. Where the draft model is right, as in copied spans, boilerplate, and code
that repeats its context, every extra drafted token is a target call saved. Where
it is wrong, every token drafted beyond the first mismatch is draft compute thrown
away, and a long draft makes each round slower than plain decoding. Real text
alternates between the two within a single response, so engines adapt: Hugging Face
`transformers` adjusts `num_assistant_tokens` with exactly this rule and stops a
draft when the assistant's confidence falls under `assistant_confidence_threshold`;
llama.cpp ends a draft at a minimum draft probability. The rules are tiny, and their
bugs do not change the generated text at all, only its cost, which is why they
survive: a policy that compares with `k` instead of the drafted count, or that never
shrinks, still decodes correctly and simply gives back the speedup it was built for.
