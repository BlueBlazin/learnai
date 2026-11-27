Compute the DRY ("don't repeat yourself") repetition penalties for the next token.
Any token that would continue a sequence already present in the recent history
receives a penalty that grows exponentially with the length of the repeated sequence.
Because this runs on every decoding step, including when text has started to loop, a
naive quadratic scan is too slow and the matching must run in linear time.

## Specification

```python
def dry_penalties(history, multiplier, base, allowed_length, breakers, last_n=None): ...
```

- `history`: a list of integer token IDs representing the context so far (prompt and
  generated tokens), oldest first. It may be empty. Do not modify it.
- `multiplier`: a float $> 0$.
- `base`: a float $\ge 1$.
- `allowed_length`: an int $\ge 1$.
- `breakers`: a set of integer token IDs acting as *sequence breakers* (such as
  newlines, quotes, colons, or chat-template markers). It may be empty.
- `last_n`: `None`, or an int $\ge 1$. When an int is given, only the **window**
  consisting of the last `last_n` tokens of `history` is visible to this function.
  When `last_n` is `None`, or when `history` has fewer than `last_n` tokens, the
  window is the entire `history`.
- Returns a `dict` mapping each penalized token ID to its penalty, a positive float
  that a sampler will subtract from that token's logit. Tokens with no penalty must
  be omitted from the dict.

Let $w_0, \ldots, w_{n-1}$ be the window of length $n$.

**Match length.** For each position $i$ with $0 \le i \le n - 2$, the match length
$m_i$ is the largest integer $m$ such that the $m$ tokens ending at position $i$
equal the $m$ tokens ending at the end of the window, and none of those tokens is a
breaker:

$$
w_{i-j} = w_{n-1-j} \quad\text{and}\quad w_{n-1-j} \notin \text{breakers}
\qquad \text{for all } 0 \le j < m .
$$

The match ending at $i$ cannot start before index $0$ of the window, so
$m_i \le i + 1$. It is allowed to overlap the suffix at the end of the window. For
example, in `[7, 7, 7]`, $m_1 = 2$.

**Candidates.** Whenever $m_i \ge$ `allowed_length`, the token $w_{i+1}$ that
immediately followed the earlier occurrence is a *candidate*, because emitting it now
would extend a repeat of $m_i$ tokens to $m_i + 1$. If a token is a candidate at
multiple positions, its effective match length $M$ is the **largest** $m_i$ among
them. Any candidate that is in `breakers` is ignored, because breakers are never
penalized.

**Penalty.** For a candidate with effective match length $M$,

$$
\text{penalty} = \text{multiplier} \cdot \text{base}^{\,\min(M - \text{allowed\_length},\; e_{\max})},
\qquad
e_{\max} = \left\lfloor \frac{88.7228391}{\ln(\text{base})} \right\rfloor .
$$

The match length $M$ counts only the tokens already in the window, not the candidate
itself. The exponent clamp $e_{\max}$ keeps the power below the largest float32
value ($88.7228391$ is its natural logarithm). Apply this clamp only when
`base > 1.000001`; for smaller bases, do not clamp the exponent.

> [!NOTE]
> A breaker stops a match wherever it appears in the trailing run of the window, not
> only when it is the final token. If the final token is a breaker, no match exists
> and the function returns `{}`.

## Example

Write `a b c y` for the token IDs `1 2 3 9`.

```python
history = [1, 2, 3, 3, 2, 3, 9, 1, 2, 3]  # a b c c b c y a b c

dry_penalties(history, 0.8, 1.75, 2, set())  # {3: 1.4, 9: 0.8}
```

| position $i$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| token $w_i$ | a | b | c | c | b | c | y | a | b |
| match length $m_i$ | 0 | 0 | 3 | 1 | 0 | 2 | 0 | 0 | 0 |
| next token $w_{i+1}$ | b | c | c | b | c | y | a | b | c |

The window ends in `a b c`. Position 2 ends an earlier `a b c` ($m_2 = 3$), followed
by `c`, so emitting `c` would repeat `a b c c`. Position 5 ends `b c` ($m_5 = 2$,
since the token before it is `c` rather than `a`), followed by `y`. Position 3 ends a
lone `c` ($m_3 = 1$), followed by `b`. Because $1 <$ `allowed_length = 2`, `b` is not
a candidate.

- `c`: $0.8 \times 1.75^{3-2} = 1.4$
- `y`: $0.8 \times 1.75^{2-2} = 0.8$

![The window ends in `a b c`. The earlier `a b c` and `b c` are followed by `c` and `y`, the two penalized tokens.](assets/history.svg)

With breakers or a shorter window on the same history:

```python
dry_penalties(history, 0.8, 1.75, 2, {2})  # {}        only `c` follows the last `b`
dry_penalties(history, 0.8, 1.75, 2, {1})  # {3: 0.8, 9: 0.8}   matches stop at `b c`
dry_penalties(history, 0.8, 1.75, 2, {9})  # {3: 1.4}  `y` is a breaker: not penalized
dry_penalties(history, 0.8, 1.75, 2, set(), 6)  # {9: 0.8}  the window is `b c y a b c`
```

In the last call, the window no longer includes the first `a b c`, so the match of
length 3 disappears and `c` is no longer a candidate. The window still contains
`b c` at positions 0 and 1 as well as at its end, so $m_1 = 2$ and the following
token `y` keeps its penalty.

## Constraints

- $0 \le \text{len}(\text{history}) \le 20{,}000$, and token IDs are non-negative
  integers.
- $0 < \text{multiplier} \le 10$, $1 \le \text{base} \le 4$, and
  $1 \le \text{allowed\_length} \le 8$.
- Output dictionary keys are compared exactly, and penalties are compared with a
  relative tolerance of $10^{-9}$ as Python floats.
- **Speed.** On a history of 20,000 tokens ending in a loop of period 20, your
  function must run within **5x** of the linear-time reference. On that input, the
  match length at every 20th position is nearly the full distance to the start of the
  loop. A naive backward scan performs about $n^2 / 40 = 10^7$ comparisons and runs
  more than 100x slower than the reference.

## Why this matters

Frequency and presence penalties discourage any token that has appeared before, which
hurts tokens that long text legitimately needs, such as names, punctuation, and
function words. Looping models do not merely repeat individual tokens. They repeat
*sequences*. DRY targets sequence repetition directly by penalizing a token only when
it would extend a verbatim repeat, growing the penalty fast enough to break a loop
within a few tokens.

DRY ships in text-generation-webui, koboldcpp, and llama.cpp, where it runs on every
decoding step over thousands of context tokens. A quadratic implementation looks fine
on short chats, but once a long generation starts to loop—the exact moment DRY is
needed—it can become slower than the model's forward pass.
