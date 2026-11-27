Implement the adaptive draft-length policy used in Hugging Face's assisted generation, combining an additive round-to-round update with an early-stop confidence check. Then simulate the policy over recorded acceptance traces and compute the total decoding cost.

![This problem is Adapt k, which sets the draft length of the next round.](assets/map.svg)

## Specification

```python
def should_stop_drafting(logits, threshold): ...  # -> bool
def next_k(k, n_accepted, n_drafted): ...  # -> int
def simulate(trace, k, schedule, threshold, c_draft, c_target): ...  # -> dict
```

### `should_stop_drafting(logits, threshold)`

`logits` is a non-empty list of floats giving the draft model's next-token logits for the token it has just drafted, and `threshold` is a float in $[0, 1]$. Let $\ell_v$ denote the logit for token $v$ in `logits`. Because the draft model decodes greedily, it picks the highest-logit token, and its *confidence* in that token is its softmax probability:

$$
\text{confidence} = \max_v \frac{e^{\,\ell_v}}{\sum_u e^{\,\ell_u}} .
$$

Return `True` if the confidence is **strictly below** `threshold`, and `False` otherwise. A threshold of `0.0` never stops drafting. Logit values can be as large as $\pm 10^4$.

### `next_k(k, n_accepted, n_drafted)`

After a round that drafted `n_drafted >= 1` tokens with current draft length `k >= 1`, the target model accepts the first `n_accepted` tokens. The draft length for the next round is

$$
\text{next\_k} = \begin{cases}
k + 2 & \text{if } n_\text{accepted} = n_\text{drafted}, \\
\max(1,\ k - 1) & \text{otherwise.}
\end{cases}
$$

Compare `n_accepted` against `n_drafted`, the number of tokens **actually drafted** in the round, which can be smaller than `k`. There is no upper limit on `k`.

### `simulate(trace, k, schedule, threshold, c_draft, c_target)`

Simulate speculative decoding over `trace`, a list of length `T = len(trace)` describing the generation of `T` output tokens. Each entry `trace[t]` is a pair `(match, logits)` for output position `t`:

- `match` (`bool`): whether the draft model's token at position `t` equals the target model's token, assuming all earlier positions are correct.
- `logits` (`list[float]`): the draft model's next-token logits at position `t`, in the same format as `should_stop_drafting`.

The remaining arguments are:

- `k` (`int >= 1`): the initial draft length.
- `schedule` (`str`): either `"constant"` or `"heuristic"`.
- `threshold` (`float` in $[0, 1]$): the confidence stopping threshold.
- `c_draft`, `c_target` (positive `int`s): the cost of one draft step and one target call.

During a round, the simulation reads `logits` at each drafted position from `trace` even if an earlier drafted token in the same round did not match.

Start at `pos = 0` with the initial `k` and repeat rounds until `pos == T`:

1. **Limit.** Set `limit = min(k, T - pos - 1)`. Because the target model always contributes one token per round, at most `T - pos - 1` drafted tokens can be used.
2. **Draft.** Draft tokens for positions `pos, pos + 1, ...` one at a time. After drafting each token, stop if `limit` tokens have been drafted or if `should_stop_drafting(logits, threshold)` returns `True` for that token's logits. The token that triggers the stop **is included in the draft**. Let `n_drafted` be the number of tokens drafted ($0 \le n_\text{drafted} \le \text{limit}$, with `n_drafted == 0` only when `limit == 0`).
3. **Verify.** Run one target call. Set `n_accepted` to the length of the longest prefix of drafted positions whose `match` is `True`. Matches after the first mismatch do not count.
4. **Account.** Add `n_drafted * c_draft + c_target` to the total cost, and advance `pos += n_accepted + 1`.
5. **Adapt.** If `schedule == "heuristic"` and `n_drafted >= 1`, update `k = next_k(k, n_accepted, n_drafted)`. If `schedule == "constant"` or `n_drafted == 0`, keep `k` unchanged.

Return a `dict` with exactly these keys:

| key | type | value |
|---|---|---|
| `"cost"` | `int` | sum of the costs of all rounds |
| `"rounds"` | `int` | number of rounds (target calls) |
| `"draft_tokens"` | `int` | sum of `n_drafted` over all rounds |
| `"ks"` | `list[int]` | value of `k` at the start of each round, in order |

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

In round 1, the policy drafts two tokens and both match. The target adds position 2, advancing `pos` by three, and `k` grows to 4.

Round 2 starts at position 3. The draft is confident at position 3 ($0.55$) even though it is wrong, so drafting continues through position 4 to position 5. At position 5, confidence is $1/3 < 0.4$, which stops the draft after including position 5 as the third token. Verification rejects the first drafted token, so nothing is accepted, the target supplies position 3, and `k` shrinks to 3.

Round 3 drafts positions 4 and 5 and stops early at position 5 for the same reason. Position 4 is accepted, position 5 is rejected, and `k` shrinks to 2. Round 4 drafts positions 6 and 7, both match, and `k` grows back to 4. Round 5 has only one token left, so `limit` is 0, nothing is drafted, the target supplies the last token, and `k` stays 4.

Decoding all ten tokens with the target model alone would cost $10 \times 10 = 100$.

## Constraints

- $1 \le T \le 400$, and $1 \le k \le 32$.
- Each logit list has 1 to 16 entries with $|\ell_v| \le 10^4$.
- `c_draft` and `c_target` are positive integers of at most 1000.
- All outputs are compared **exactly**, including their Python types (`bool`, `int`, and a `dict` of `int`s and a `list[int]`). In every test, each position's confidence differs from `threshold` by at least $10^{-6}$, except in one strict-inequality test where all logits are equal and the confidence is exactly $1/2$ or $1/4$.
- **Cost.** On traces that alternate between phases where the draft is almost always right and phases where it is usually wrong and unsure, `simulate` with `schedule="heuristic"`, `threshold=0.4`, `c_draft=1`, and `c_target=10` must achieve a cost of at most $0.9$ times the best constant schedule without confidence stopping (the minimum over $k = 1, \dots, 32$ of `simulate(trace, k, "constant", 0.0, 1, 10)`, computed by the judge and rounded down). The specified policy achieves about $0.8$ to $0.85$ of that baseline on those traces.
- Only the Python standard library is needed.

## Why this matters

No single constant draft length works well across an entire response. On easy spans such as copied text, boilerplate, or repetitive code, every extra drafted token saves an expensive target call. On hard spans where the draft model is wrong, every token drafted past the first mismatch wastes draft compute and can make speculative decoding slower than plain decoding.

Production engines adapt dynamically because real text alternates between easy and hard spans within a single generation. Hugging Face `transformers` adjusts `num_assistant_tokens` with this exact additive rule and stops early using `assistant_confidence_threshold`, while `llama.cpp` stops drafting below a minimum draft probability. Bugs in these small rules never change the generated text, only the latency and compute cost. A policy that compares `n_accepted` against `k` instead of `n_drafted`, or that never shrinks below the initial `k`, still decodes correctly while quietly throwing away its speedup.
