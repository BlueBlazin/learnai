Compute the DRY ("don't repeat yourself") penalties for the next token: every token
that would continue a sequence the history already contains gets a penalty that
grows exponentially with the length of the repeat. Deciding which tokens those are
is a string-matching problem, and on the input the sampler exists for, text that has
started to loop, the obvious scan is quadratic.

## Specification

```python
def dry_penalties(history, multiplier, base, allowed_length, breakers, last_n=None): ...
```

- `history`: a list of ints, the token ids of the context so far (prompt and
  generated tokens), oldest first. It may be empty. Do not modify it.
- `multiplier`: a float $> 0$. `base`: a float $\ge 1$. `allowed_length`: an int
  $\ge 1$.
- `breakers`: a set of ints, the *sequence breakers* (in practice the tokens of
  newlines, quotes, colons, and chat-template markers). It may be empty.
- `last_n`: `None`, or an int $\ge 1$. With an int, only the **window** made of the
  last `last_n` tokens of the history exists for this function; with `None`, or when
  the history is shorter, the window is the whole history.
- Returns a dict that maps a token id to its penalty, a positive float: the amount
  the sampler will subtract from that token's logit. A token without a penalty is
  not in the dict.

Let $w_0, \ldots, w_{n-1}$ be the window.

**Match length.** For a position $i$ with $0 \le i \le n - 2$, the match length
$m_i$ is the largest $m$ such that the $m$ tokens ending at position $i$ equal the
$m$ tokens ending at the end of the window, and none of them is a breaker:

$$
w_{i-j} = w_{n-1-j} \quad\text{and}\quad w_{n-1-j} \notin \text{breakers}
\qquad \text{for all } 0 \le j < m .
$$

The run ending at $i$ cannot start before the window does, so $m_i \le i + 1$. It
may overlap the run at the end of the window: in `[7, 7, 7]`, $m_1 = 2$.

**Candidates.** If $m_i \ge$ `allowed_length`, the token $w_{i+1}$ that followed
the earlier occurrence is a *candidate*: emitting it now would extend a repeat of
$m_i$ tokens to $m_i + 1$. A token can be a candidate at several positions; its
match length $M$ is the **largest** $m_i$ among them. A candidate that is a breaker
is dropped: breakers are never penalized.

**Penalty.** For a candidate with match length $M$,

$$
\text{penalty} = \text{multiplier} \cdot \text{base}^{\,\min(M - \text{allowed\_length},\; e_{\max})},
\qquad
e_{\max} = \left\lfloor \frac{88.7228391}{\ln(\text{base})} \right\rfloor .
$$

The match length counts the tokens already in the window, not the candidate. The
clamp $e_{\max}$ keeps the power below the largest float32 ($88.7228391$ is its
natural logarithm); it applies when `base > 1.000001`, and for a smaller base the
exponent is not clamped.

> [!NOTE]
> A breaker stops a match wherever it sits in the run at the end of the window, not
> only when it is the last token. If the last token is a breaker, no match exists
> and the result is empty.

## Example

Write `a b c y` for the tokens `1 2 3 9`.

```python
history = [1, 2, 3, 3, 2, 3, 9, 1, 2, 3]  # a b c c b c y a b c

dry_penalties(history, 0.8, 1.75, 2, set())  # {3: 1.4, 9: 0.8}
```

| position $i$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| token $w_i$ | a | b | c | c | b | c | y | a | b |
| match length $m_i$ | 0 | 0 | 3 | 1 | 0 | 2 | 0 | 0 | 0 |
| next token $w_{i+1}$ | b | c | c | b | c | y | a | b | c |

The window ends in `a b c`. Position 2 ends an earlier `a b c` ($m = 3$), and the
token after it is `c`: emitting `c` would repeat `a b c c`. Position 5 ends `b c`
($m = 2$; the token before it is `c`, not `a`), followed by `y`. Position 3 ends a
lone `c` ($m = 1$), followed by `b`; that is below `allowed_length = 2`, so `b` is
not a candidate.

- `c`: $0.8 \times 1.75^{3-2} = 1.4$
- `y`: $0.8 \times 1.75^{2-2} = 0.8$

With breakers, on the same history:

```python
dry_penalties(history, 0.8, 1.75, 2, {2})  # {}        only `c` follows the last `b`
dry_penalties(history, 0.8, 1.75, 2, {1})  # {3: 0.8, 9: 0.8}   matches stop at `b c`
dry_penalties(history, 0.8, 1.75, 2, {9})  # {3: 1.4}  `y` is a breaker: not penalized
dry_penalties(history, 0.8, 1.75, 2, set(), 6)  # {9: 0.8}  the window is `b c y a b c`
```

In the last call the window no longer contains the first `a b c`, so the match of
length 3 is gone and `c` is not a candidate. The window still holds `b c` at its
positions 0 and 1 and at its end, so $m_1 = 2$ and the token after it, `y`, keeps
its penalty.

## Constraints

- $0 \le$ `len(history)` $\le 20{,}000$; token ids are non-negative ints.
  $0 < \text{multiplier} \le 10$, $1 \le \text{base} \le 4$,
  $1 \le \text{allowed\_length} \le 8$.
- The keys of the result are compared exactly, the penalties with a relative
  tolerance of $10^{-9}$ (they are Python floats).
- **Speed.** On a history of 20,000 tokens that ends in a loop with a period of 20
  tokens, the function must run within **5x** of the reference, which is linear in
  the length of the window. There the match length at every 20th position is nearly
  the distance to the start of the loop, so comparing token by token from every
  position does about $n^2 / 40 = 10^7$ comparisons; measured, that is more than
  100x slower than the reference.

## Why this matters

Frequency and presence penalties punish a token for having appeared, which hurts
exactly the tokens a long text needs most: names, punctuation, function words.
Models that loop do not repeat tokens, they repeat *sequences*, and DRY penalizes
only that: a token is penalized only when it would extend a verbatim repeat, and
the penalty grows fast enough to break a loop within a few tokens. It ships in
text-generation-webui, koboldcpp, and llama.cpp, and it runs on every decoding step
over a history of thousands of tokens. A sampler step that is quadratic in the
context length is invisible in a short chat and takes longer than the model's
forward pass once a long generation starts to loop, which is the moment the sampler
is needed.
