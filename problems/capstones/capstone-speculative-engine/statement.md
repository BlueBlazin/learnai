Build a speculative decoding engine in five stages: a greedy loop that drafts with a
small model, verifies with one call of the large one, and rolls both KV caches back
after every rejection; sampling that stays exactly on the target's distribution; a
draft length that adapts to how often the draft is right; prompt lookup as a second,
free proposer; and a batch whose rows accept different numbers of tokens. Each stage
is an earlier problem; the capstone is that they share one loop, two caches, and one
stream of supplied randomness, and that every shortcut between them changes either
the output or the bill.

## Overview

```python
class SpeculativeEngine:
    def __init__(self, target, draft, *, k=4, adaptive=False, lookup=None): ...

    rounds: list  # [(proposer, n_drafted, n_accepted), ...] of the last generate call

    def generate(self, prompt, max_new_tokens, uniforms=None): ...  # stages 1 to 4
    def generate_batch(self, prompts, max_new_tokens): ...  # stage 5
```

![One round. A proposer (prompt lookup if it finds a match, otherwise the draft model through its cache) hands up to $k$ draft tokens to the target, which scores them all in one call. The verifier accepts a prefix and adds one token of the target's own; then both caches are rolled back to verified tokens and $k$ is adjusted.](assets/engine.svg)

The cases are grouped by stage, and each stage's cases use only what that stage and
the ones before it specify: `uniforms` is `None` until stage 2, `adaptive` is `False`
until stage 3, `lookup` is `None` until stage 4, and `generate_batch` is called only
in stage 5.

**The models.** `target` and `draft` are two
[`TinyLM`](../../../python/src/learnai/kit/tinylm.py) models over one vocabulary,
built by [`draft_target_pair`](../../../python/src/learnai/kit/draft.py):

```python
logits, cache = lm(tokens)  # no cache: `tokens` is the whole sequence so far
logits, cache = lm(tokens, cache)  # with a cache: `tokens` continue the cached prefix
short = cache.truncate(n)  # a new cache covering only the first n tokens
```

`tokens` is a list of ints or an integer tensor of shape `(T,)`, $T \ge 1$; `logits`
is float32 of shape `(T, vocab_size)`, and row `i` scores the token after
`tokens[i]`. Caches are immutable: a call or a `truncate` returns a new cache
(`cache.length` tokens long). Each model counts `forward_calls` and
`tokens_processed` in its own `counters`. The models are causal. You may use
`ops.softmax` from the kit's
[`ops`](../../../python/src/learnai/kit/ops/__init__.py).

**A round.** Write $S$ for the verified sequence (the prompt followed by the tokens
emitted so far), $r$ for the number of tokens still to emit, and $k$ for the current
draft length. `generate` repeats this until $r = 0$:

1. **Limit.** $\ell = \min(k,\ r - 1)$: a round emits up to $\ell + 1$ tokens.
2. **Propose** $m \le \ell$ draft tokens $x_0, \dots, x_{m-1}$ that continue $S$.
   Until stage 4 the proposer is the draft model and $m = \ell$.
3. **Verify.** **One** target call yields the target's logits at $m + 1$ positions:
   row $j$ scores the token after $S, x_0, \dots, x_{j-1}$.
4. **Accept** the first $n \le m$ drafts and **emit** $x_0, \dots, x_{n-1}$ followed
   by one token chosen from the target's row $n$. That is $n + 1 \le r$ tokens.
5. **Record** `(proposer, m, n)` in `rounds` (`proposer` is `"draft"` or
   `"lookup"`), adapt $k$ (stage 3), and bring both caches back to verified tokens.

`generate` returns the `max_new_tokens` emitted tokens as a list of Python ints (the
prompt is not included). Every call of `generate` starts with empty caches, the
constructor's `k`, and an empty `rounds`. There is no stop token.

## Stage 1: greedy speculation with cache rollback

With `uniforms=None` everything is an argmax (ties to the lowest id; none occur in
the tests):

- **Propose.** $x_j$ is the argmax of the draft model's logits after
  $S, x_0, \dots, x_{j-1}$, for $j = 0, \dots, \ell - 1$.
- **Accept.** With $g_j$ the argmax of the target's row $j$: $n$ is the number of
  leading drafts with $x_j = g_j$, and the round emits $x_0, \dots, x_{n-1}, g_n$:
  the target's correction at the first mismatch, or its bonus token when all drafts
  matched.

The result is exactly the target's own greedy continuation of the prompt, whatever
the draft model says.

**Cost.** Let $P$ be the prompt length, $N$ = `max_new_tokens`, $R$ the number of
rounds, $D$ the total number of drafted tokens (the sum of $m$), and $D_m$ the
number drafted by the draft model. In every stage:

| counter | limit | why |
|---|---|---|
| target `forward_calls` | $R$ | one verification per round |
| target `tokens_processed` | $P + D + R$ | the prompt once; per round, the drafts and the one verified token the target has not seen |
| draft `tokens_processed` | $P + N + D_m$ | every verified token at most once, and every token it drafted at most once more |

Feeding a model its whole sequence again exceeds the token limits. A cache that
still holds rejected drafts gives a wrong output (target) or other, usually more,
rounds than the procedure (draft): `rounds` is compared exactly.

### Example 1

```python
draft, target = draft_target_pair(fw, seed=3, agreement=0.9)
engine = SpeculativeEngine(target, draft, k=4)
engine.generate([0, 17, 42, 5], 12)
# [42, 62, 33, 33, 40, 32, 45, 37, 40, 33, 40, 55]
engine.rounds
# [("draft", 4, 0), ("draft", 4, 4), ("draft", 4, 4), ("draft", 0, 0)]
```

| round | $r$ | drafts | target's argmax per row | $n$ | emitted |
|---|---|---|---|---|---|
| 1 | 12 | `33 40 33 33` | `42` ... | 0 | `42` |
| 2 | 11 | `62 33 33 40` | `62 33 33 40 32` | 4 | `62 33 33 40 32` |
| 3 | 6 | `45 37 40 33` | `45 37 40 33 40` | 4 | `45 37 40 33 40` |
| 4 | 1 | (none) | `55` | 0 | `55` |

In round 1 the first draft is wrong; the three after it are void and both caches
hold tokens that must go. In round 4 one token is left, so nothing is drafted. The
reference feeds the target $8 + 5 + 5 + 1 = 19$ tokens in 4 calls (limit
$4 + 12 + 4 = 20$) and the draft 16 (limit $4 + 12 + 12 = 28$).

## Stage 2: sampling with exact rejection

With `uniforms`, a float32 tensor of shape `(R_max, J_max, 3)` with values in
$[0, 1)$, the engine samples, and the emitted tokens are distributed exactly as if
the target model had sampled them itself. Randomness is an input: round number
$\rho = 0, 1, \dots$ of the call uses only `uniforms[ρ]`, and each entry at most
once. Write $u_{j,c}$ for `uniforms[ρ, j, c]`. All distributions are the softmax of
the model's logits, and "sample with $u$ from weights $w$" means inverse CDF: with
$c_v = \sum_{i \le v} w_i$, the token $\min\{v : c_v > u \cdot c_{V-1}\}$.

- **Propose.** $q_j$ is the draft model's distribution after
  $S, x_0, \dots, x_{j-1}$, and $x_j$ is sampled with $u_{j,0}$ from $q_j$.
- **Accept.** $p_j$ is the target's distribution in row $j$. Draft $j$ passes when
  $u_{j,1} \cdot q_j[x_j] < p_j[x_j]$, and $n$ is the number of drafts before the
  first one that does not pass ($m$ if all pass). Tests of later drafts do not count.
- **Emit one more token**, sampled with $u_{n,2}$:
  from the residual $\max(0,\ p_n - q_n)$ if $n < m$ (a rejection), from $p_m$ itself
  if $n = m$ (the bonus token, also when $m = 0$).

This is `speculative-sampling-block` once per round. The tests pass
`R_max = J_max = max_new_tokens`, which is always enough.

### Example 2

One round with $m = 2$ over a vocabulary of 4, with the target and draft
distributions

```python
p = [[0.10, 0.40, 0.20, 0.30], [0.25, 0.25, 0.25, 0.25], [0.50, 0.10, 0.10, 0.30]]
q = [[0.50, 0.20, 0.20, 0.10], [0.10, 0.60, 0.20, 0.10]]
u = [[0.6, 0.9, 0.0], [0.3, 0.7, 0.5], ...]  # uniforms[ρ]
```

Draft 0 is sampled with $0.6$ from `q[0]` (cumulative sums `0.5, 0.7, 0.9, 1.0`):
token 1. Draft 1 is sampled with $0.3$ from `q[1]`, the draft model's distribution
after token 1 (`0.1, 0.7, 0.9, 1.0`): token 1. Draft 0 passes
($0.9 \cdot 0.2 = 0.18 < 0.4$); draft 1 does not ($0.7 \cdot 0.6 = 0.42 \ge 0.25$),
so $n = 1$. The residual of position 1 is `[0.15, 0, 0.05, 0.15]` with total 0.35;
$u_{1,2} = 0.5$ gives the threshold $0.175$, and the first cumulative sum above it
(`0.15, 0.15, 0.20, 0.35`) belongs to token 2. The round emits `1, 2` and records
`("draft", 2, 1)`.

## Stage 3: adaptive draft length

With `adaptive=True`, after every round that drafted at least one token ($m \ge 1$):

$$
k \leftarrow \begin{cases} k + 2 & \text{if } n = m \quad \text{(every drafted token was accepted)}, \\ \max(1,\ k - 1) & \text{otherwise.} \end{cases}
$$

The comparison is with the number of tokens actually drafted, which can be smaller
than $k$. A round with $m = 0$ leaves $k$ unchanged. This is the rule of
`dynamic-draft-length` (without its confidence stop), in greedy and in sampling
mode alike. The limits of stage 1 apply with the $R$, $D$, and $D_m$ of the adaptive
procedure.

### Example 3

`draft_target_pair(fw, seed=3, agreement=0.95)`, prompt `[0, 10, 23, 33]`, 48 tokens,
`k=4, adaptive=True`: the rounds draft `4, 6, 8, 7, 9, 9, 8` tokens and accept
`4, 6, 7, 7, 9, 0, 8`, so the output takes **7** target calls; with $k = 4$
throughout it takes 10. With `seed=4, agreement=0.5` and prompt `[0, 11, 24, 33]`,
$k$ falls to 1 within four rounds; the 48 tokens take 29 target calls either way,
but the reference's draft model processes 75 tokens instead of 111 and its target
99 instead of 138.

## Stage 4: prompt lookup as a second proposer

With `lookup=(max_ngram, min_ngram)`, a round with $\ell \ge 1$ first tries to copy
its drafts from the sequence itself, as in `prompt-lookup-draft`. Let $L$ be the
length of $S$. For $n_g =$ `max_ngram`, `max_ngram - 1`, ..., `min_ngram`, in that
order, look for an **earlier occurrence** of the last $n_g$ tokens of $S$: a start
index $s$ with $0 \le s < L - n_g$ and $S[s : s + n_g] = S[L - n_g : L]$
(occurrences may overlap the suffix; lengths $n_g \ge L$ are skipped). Stop at the
first $n_g$ that has one, take the **earliest** $s$, and copy

$$
S[s + n_g : s + n_g + \ell],
$$

which is between 1 and $\ell$ tokens.

- If the lookup finds tokens, they are the round's drafts: the proposer is
  `"lookup"`, $m$ is their number, and the draft model is **not called** in this
  round.
- Otherwise the draft model proposes as before, and the proposer is `"draft"`.

Copied tokens are proposals like any other and are verified by the target:

- greedy mode: exactly the acceptance rule of stage 1;
- sampling mode: a copied token was proposed with probability 1, so $q_j$ is the
  distribution that puts all its mass on $x_j$. The rules of stage 2 then read:
  draft $j$ passes when $u_{j,1} < p_j[x_j]$, and after a rejection the residual is
  $p_n$ with the entry of $x_n$ set to zero. $u_{j,0}$ is not used.

The draft model's cache simply falls behind during lookup rounds and catches up, in
one call, the next time the draft model proposes. The adaptive rule of stage 3
treats both proposers alike.

### Example 4

`draft_target_pair(fw, seed=36, agreement=0.6)`, prompt `[0, 5, 5, 9]`, 60 tokens,
`k=4, lookup=(3, 2)`. The target's greedy output is `51, 44` followed by token 48
repeated, with one more 44. The rounds are

```python
[
    ("draft", 4, 0),
    ("draft", 4, 4),
    ("lookup", 1, 1),
    ("lookup", 3, 3),
    ("draft", 4, 4),
    ("lookup", 4, 4),
    ...,
]  # seven more full lookup rounds, then ("lookup", 2, 2)
```

14 target calls, and the draft model processes 22 tokens; without lookup the same
output takes 15 target calls and 68 draft tokens. Lookup is free but not always
right: on the second public fixture the target rejects copied tokens in 12 of 17
lookup rounds.

## Stage 5: a batch with ragged acceptance

`generate_batch(prompts, max_new_tokens)` decodes $B$ prompts of equal length $P$
greedily, with the draft model as the only proposer and the constructor's `k`
(`adaptive` and `lookup` do not apply). Each row runs the rounds of stage 1 on its
own: row $b$ of the result equals `generate(prompts[b], max_new_tokens)`, and row
$b$ needs the same $R_b$ rounds it would need alone. It returns a list of $B$ lists
of Python ints.

What is shared is the target call. The kit models accept a batch: `tokens` of shape
`(B, T)` gives `logits` of shape `(B, T, vocab_size)`, counts as **one** forward
call and $B \cdot T$ tokens, and returns a cache that covers the **same** number of
tokens for every row. `cache.truncate(n)` cuts all rows to `n` tokens;
`cache.select(rows)` keeps the listed rows (in that order). Because the models are
causal, tokens appended to the right of a row do not change the logits before them,
so a shorter row can be padded on the right with any token id.

**Cost.**

- target `forward_calls` $\le \max_b R_b$: one call per round for the whole batch,
  and no row waits for another. A batch that keeps only as many drafts as its
  unluckiest row needs more rounds; rows decoded one after another need
  $\sum_b R_b$ calls.
- target `tokens_processed` $\le 1.25\,T$, where $T$ is what this procedure
  processes. Let $c$ be the number of leading tokens of every active row that the
  cache holds ($c = 0$ at first). In a round, row $b$ with $\text{len}_b$ verified
  tokens feeds $\text{len}_b - c + m_b$ tokens (its verified tokens from $c$ on, then
  its drafts); the call is as wide as the widest row and costs (active rows)
  $\times$ (width). Afterwards the entries of row $b$ are valid for its first
  $\text{len}_b + n_b$ tokens. Rows that have all their tokens leave the batch, and
  $c$ becomes the smallest valid length among the rows that remain.
- draft `tokens_processed` $\le \sum_b (P + N + D_b)$, the limit of stage 1 per row.
  The draft model may be called per row.

### Example 5

`draft_target_pair(fw, seed=4, agreement=0.5)`, `k=4`, 40 tokens for each of
`[0, 23, 13, 42, 38, 9]` and `[0, 57, 42, 51, 25, 13]`. Alone, the rows need 21 and
19 rounds; in their first eight rounds they accept `0, 1, 0, 0, 0, 0, 4, 1` and
`1, 4, 4, 0, 0, 3, 0, 2` drafts. The batch must finish in **21** target calls; the
reference's target processes 405 tokens (limit 506). Cutting both rows to the
shorter acceptance of each round gives the same tokens in 31 calls, and decoding
the rows one after the other takes 40.

## Constraints

- $1 \le P$, $1 \le k \le 8$ at construction, $1 \le$ `max_new_tokens`, and
  $P +$ `max_new_tokens` $\le 128$, the context of both models. `min_ngram` $\ge 1$,
  `max_ngram` $\le 8$. $1 \le B \le 4$.
- **Greedy runs** (stages 1, 3, 4, 5): tokens are compared exactly with the
  target's greedy decode and `rounds` with the procedure's. Wherever a model's
  argmax is used, its two largest logits differ by at least `1e-3`, so the way calls
  are split (which changes logits by about `1e-5`) affects nothing.
- **Sampling runs with exact comparison** (stages 2 to 4): the supplied uniforms keep
  every decision away from its boundary: $|u \cdot q_j[x_j] - p_j[x_j]| \ge 10^{-3}
  \max(u \cdot q_j[x_j],\ p_j[x_j])$ for every examined draft, and
  $|u \cdot c_{V-1} - c_v| \ge 10^{-4}\, c_{V-1}$ for every inverse-CDF sample and
  every $v$. Tokens and `rounds` are compared exactly.
- **Statistical cases** (stages 2 and 4): with a pair of 8-token models, the test
  makes 600 calls with independent uniforms, each generating 3 tokens, and compares
  the distribution of each of the three tokens, and of the first two jointly, with
  the exact distribution of the target's own samples, using finite-sample Chernoff
  bounds. A correct solution fails with probability below $10^{-9}$, and since the
  seeds are fixed, it either always passes or always fails.
- **Cost**: the limits of stages 1 and 5 are checked in every case of their kind,
  after the tokens. They are exact counts.
- Do not draw random numbers yourself. `rounds` may hold tuples or lists.
- The time limit per case is 30 seconds. The statistical cases call `generate`
  600 times: keep the tensor work per round to a handful of operations.

## Why this matters

This is the speculative path of a real server: a draft model beside the main model
in llama.cpp, the draft-model and n-gram speculators of vLLM, assisted generation
and prompt lookup in `transformers`. The single-step rule fits on a line. What
ships is the loop around it: two caches that advance at different speeds and must
both be rewound; a sampler whose correctness depends on which uniform decides what;
a draft length that pays for itself only if it follows the acceptance rate; a
proposer that costs nothing and is wrong half the time; and a batch whose rows
refuse to stay the same length. Every one of these fails quietly. A stale target
cache yields fluent text that is no longer the target's. A residual sampled without
normalizing, or a copied token trusted without verification, moves the output
distribution toward the proposer, and no single response shows it. A batch cut to
its unluckiest row returns exactly the right tokens at a fraction of the promised
speedup, which only a benchmark reveals.
