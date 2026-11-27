Implement `SpeculativeEngine`, a five-stage speculative decoding engine that combines a draft model, a target model, KV cache rollback, exact rejection sampling, adaptive draft length, prompt lookup, and ragged-acceptance batching in one loop.

## Overview

```python
class SpeculativeEngine:
    def __init__(self, target, draft, *, k=4, adaptive=False, lookup=None): ...

    rounds: list  # [(proposer, n_drafted, n_accepted), ...] of the last generate call

    def generate(self, prompt, max_new_tokens, uniforms=None): ...  # stages 1 to 4
    def generate_batch(self, prompts, max_new_tokens): ...  # stage 5
```

![One round. A proposer drafts up to $k$ tokens, the target scores them in one call, and the verifier accepts a prefix.](assets/engine.svg)

The test suite groups cases by stage, and each stage uses only the features introduced up to that point. `uniforms` is `None` until stage 2, `adaptive` is `False` until stage 3, `lookup` is `None` until stage 4, and `generate_batch` is called only in stage 5.

**The models.** `target` and `draft` are two [`TinyLM`](../../../python/src/learnai/kit/tinylm.py) models over a shared vocabulary of size $V = \text{vocab\_size}$, built by [`draft_target_pair`](../../../python/src/learnai/kit/draft.py).

```python
logits, cache = lm(tokens)  # no cache: `tokens` is the whole sequence so far
logits, cache = lm(tokens, cache)  # with a cache: `tokens` continue the cached prefix
short = cache.truncate(n)  # a new cache covering only the first n tokens
```

`tokens` is a list of ints or an integer tensor of shape `(T,)` for sequence length $T \ge 1$. `logits` is a float32 tensor of shape `(T, V)`, where row `i` scores the token immediately after `tokens[i]`. Caches are immutable, so every model call or `truncate` returns a new cache whose length is `cache.length`. Each model tracks `forward_calls` and `tokens_processed` in its `counters` attribute. Both models are causal. You may use `ops.softmax` from the kit's [`ops`](../../../python/src/learnai/kit/ops/__init__.py).

**A round.** Let $S$ be the verified sequence (the prompt followed by all tokens emitted so far), $r$ be the number of tokens still to emit, and $k$ be the current draft length. `generate` repeats these five steps until $r = 0$:

1. **Limit.** Set $\ell = \min(k,\ r - 1)$. Because a round emits at most $\ell + 1$ tokens, this prevents overshooting `max_new_tokens`.
2. **Propose.** Generate $m \le \ell$ draft tokens $x_0, \dots, x_{m-1}$ continuing $S$. Until stage 4, the draft model is the only proposer and $m = \ell$.
3. **Verify.** Run **one** target forward call to obtain target logits at $m + 1$ positions, where row $j$ scores the token after $S, x_0, \dots, x_{j-1}$.
4. **Accept.** Accept the first $n \le m$ drafts and emit $x_0, \dots, x_{n-1}$ followed by one token chosen from the target's row $n$. This adds $n + 1 \le r$ tokens to $S$.
5. **Record and roll back.** Append `(proposer, m, n)` to `rounds` (where `proposer` is `"draft"` or `"lookup"`), update $k$ if adaptive mode is on (stage 3), and roll both caches back so they hold only verified tokens.

`generate` returns the `max_new_tokens` emitted tokens as a list of Python ints, excluding the prompt. Every call to `generate` starts with empty caches, the constructor's `k`, and an empty `rounds` list. There is no stop token.

## Stage 1: greedy speculation with cache rollback

When `uniforms` is `None`, all token selections use argmax (breaking ties toward the lowest token id, though no ties occur in the tests).

- **Propose.** For $j = 0, \dots, \ell - 1$, draft token $x_j$ is the argmax of the draft model's logits after $S, x_0, \dots, x_{j-1}$.
- **Accept.** Let $g_j$ be the argmax of the target's row $j$. Let $n$ be the number of leading drafts where $x_j = g_j$. The round emits $x_0, \dots, x_{n-1}, g_n$, which is either the target's correction at the first mismatch or its bonus token when all drafts match.

The output always matches the target model's own greedy continuation of the prompt, regardless of what the draft model predicts.

**Cost.** Let $P$ be the prompt length, $N = \text{max\_new\_tokens}$, $R$ be the number of rounds, $D$ be the total number of drafted tokens (the sum of $m$ across rounds), and $D_m$ be the number of tokens drafted by the draft model. Every stage must stay within these limits:

| counter | limit | why |
|---|---|---|
| target `forward_calls` | $R$ | one verification per round |
| target `tokens_processed` | $P + D + R$ | the prompt once, plus each round's drafts and the one verified token the target has not seen |
| draft `tokens_processed` | $P + N + D_m$ | every verified token at most once, plus every token it drafted at most once more |

Refeeding a model its entire sequence exceeds these token limits. Leaving rejected drafts in the target cache produces wrong output tokens. Leaving rejected drafts in the draft cache changes the draft proposals and produces a different `rounds` trace, which is compared exactly.

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

In round 1, the first draft is wrong, so the remaining three drafts are discarded and both caches must drop them. In round 4, only one token remains, so nothing is drafted. The reference feeds the target $8 + 5 + 5 + 1 = 19$ tokens in 4 calls (limit $4 + 12 + 4 = 20$) and feeds the draft 16 tokens (limit $4 + 12 + 12 = 28$).

## Stage 2: sampling with exact rejection

When `uniforms` is provided, it is a float32 tensor of shape `(R_max, J_max, 3)` with values in $[0, 1)$, where `R_max` is at least the number of rounds and `J_max` is at least the maximum drafts per round. The engine samples from the models so that the emitted tokens follow the exact distribution of the target model.

All randomness comes from `uniforms`. Round $\rho = 0, 1, \dots$ uses only `uniforms[ρ]`, and each entry is used at most once. Write $u_{j,c}$ for `uniforms[ρ, j, c]`. Every token distribution is the softmax of the model's logits. To sample with $u$ from unnormalized weights $w$ of length $V$, compute the cumulative sums $c_v = \sum_{i \le v} w_i$ and return the smallest token id $v$ satisfying $c_v > u \cdot c_{V-1}$.

- **Propose.** Let $q_j$ be the draft model's distribution after $S, x_0, \dots, x_{j-1}$. Sample $x_j$ from $q_j$ using $u_{j,0}$.
- **Accept.** Let $p_j$ be the target's distribution in row $j$. Draft $j$ passes if $u_{j,1} \cdot q_j[x_j] < p_j[x_j]$. Set $n$ to the number of drafts before the first draft that fails (or $m$ if all pass). Later drafts are ignored once a failure occurs.
- **Emit one more token.** Sample the extra token using $u_{n,2}$. If $n < m$ (a rejection), sample from the residual weights $\max(0,\ p_n - q_n)$. If $n = m$ (all drafts accepted, including when $m = 0$), sample from $p_m$.

This applies the `speculative-sampling-block` rule once per round. The tests pass `R_max = J_max = max_new_tokens`, which is always large enough.

### Example 2

Consider one round with $m = 2$ over a vocabulary of size $V = 4$, with target distributions `p`, draft distributions `q`, and round uniforms `u = uniforms[ρ]`:

```python
p = [[0.10, 0.40, 0.20, 0.30], [0.25, 0.25, 0.25, 0.25], [0.50, 0.10, 0.10, 0.30]]
q = [[0.50, 0.20, 0.20, 0.10], [0.10, 0.60, 0.20, 0.10]]
u = [[0.6, 0.9, 0.0], [0.3, 0.7, 0.5], ...]  # uniforms[ρ]
```

1. Draft 0 is sampled from `q[0]` (cumulative sums `0.5, 0.7, 0.9, 1.0`) with $u_{0,0} = 0.6$, giving token 1.
2. Draft 1 is sampled from `q[1]` (the draft distribution after token 1, with cumulative sums `0.1, 0.7, 0.9, 1.0`) with $u_{1,0} = 0.3$, giving token 1.
3. Draft 0 passes because $0.9 \cdot 0.2 = 0.18 < 0.4$. Draft 1 fails because $0.7 \cdot 0.6 = 0.42 \ge 0.25$, so $n = 1$.
4. The residual at position 1 is `[0.15, 0, 0.05, 0.15]`, which sums to 0.35. With $u_{1,2} = 0.5$, the threshold is $0.5 \cdot 0.35 = 0.175$. The first cumulative sum above 0.175 (`0.15, 0.15, 0.20, 0.35`) is at token 2.

The round emits `1, 2` and records `("draft", 2, 1)`.

## Stage 3: adaptive draft length

When `adaptive=True`, update $k$ after every round that drafted at least one token ($m \ge 1$) to grow $k$ on full acceptance and shrink it on rejection:

$$
k \leftarrow \begin{cases} k + 2 & \text{if } n = m \quad \text{(every drafted token was accepted)}, \\ \max(1,\ k - 1) & \text{otherwise.} \end{cases}
$$

Compare $n$ against $m$, the number of tokens actually drafted in that round, which can be smaller than $k$. A round with $m = 0$ leaves $k$ unchanged. This matches the heuristic schedule of `dynamic-draft-length` (without its confidence stop) and applies to both greedy and sampling modes. The stage 1 cost limits apply using the $R$, $D$, and $D_m$ produced by this adaptive loop.

### Example 3

With `draft_target_pair(fw, seed=3, agreement=0.95)`, prompt `[0, 10, 23, 33]`, 48 tokens, and `k=4, adaptive=True`, the rounds draft `4, 6, 8, 7, 9, 9, 8` tokens and accept `4, 6, 7, 7, 9, 0, 8`. Generating the output takes **7** target calls instead of 10 with a fixed $k = 4$.

With `seed=4, agreement=0.5` and prompt `[0, 11, 24, 33]`, $k$ drops to 1 within four rounds. Generating 48 tokens takes 29 target calls either way, but the reference's draft model processes 75 tokens instead of 111 and its target processes 99 instead of 138.

## Stage 4: prompt lookup as a second proposer

When `lookup=(max_ngram, min_ngram)` is set and $\ell \ge 1$, each round first tries to copy draft tokens from earlier in the verified sequence $S$, as in `prompt-lookup-draft`. Let $L$ be the length of $S$. For $n_g = \text{max\_ngram}, \text{max\_ngram} - 1, \dots, \text{min\_ngram}$ in descending order, search for an earlier occurrence of the last $n_g$ tokens of $S$, meaning a start index $s$ with $0 \le s < L - n_g$ and $S[s : s + n_g] = S[L - n_g : L]$. Occurrences may overlap the suffix, and any $n_g \ge L$ is skipped. Stop at the first $n_g$ that matches, pick the **earliest** matching index $s$, and copy up to $\ell$ tokens immediately following it:

$$
S[s + n_g : s + n_g + \ell].
$$

This yields between 1 and $\ell$ draft tokens.

- If the lookup finds a match, those copied tokens are the round's drafts, `proposer` is `"lookup"`, $m$ is the number of copied tokens, and the draft model is **not called** in this round.
- Otherwise, the draft model proposes as usual and `proposer` is `"draft"`.

Copied tokens are verified by the target just like model drafts:

- In greedy mode, apply the exact acceptance rule from stage 1.
- In sampling mode, a copied token is proposed deterministically, so $q_j$ puts probability 1 on $x_j$ and 0 everywhere else. Under the stage 2 formulas, draft $j$ passes when $u_{j,1} < p_j[x_j]$, and on rejection the residual is $p_n$ with entry $x_n$ set to 0. $u_{j,0}$ is not used.

During lookup rounds, the draft model's cache simply stays behind and catches up in one forward call the next time the draft model proposes. The adaptive rule of stage 3 applies to both `"draft"` and `"lookup"` rounds identically.

### Example 4

With `draft_target_pair(fw, seed=36, agreement=0.6)`, prompt `[0, 5, 5, 9]`, 60 tokens, and `k=4, lookup=(3, 2)`, the target's greedy output is `51, 44` followed by repeated `48`s and one more `44`. The rounds are:

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

This takes 14 target calls and 22 draft tokens, compared to 15 target calls and 68 draft tokens without lookup. Lookup is free to propose, but its guesses are not guaranteed to match. On the second public fixture, the target rejects copied tokens in 12 of 17 lookup rounds.

## Stage 5: a batch with ragged acceptance

`generate_batch(prompts, max_new_tokens)` greedily decodes a batch of $B$ prompts of equal length $P$ using only the draft model and the constructor's `k` (`adaptive` and `lookup` do not apply). Each row $b$ runs the exact same stage 1 rounds as it would alone, producing the same tokens as `generate(prompts[b], max_new_tokens)` in the same number of rounds $R_b$. The method returns a list of $B$ lists of Python ints.

The rows share each target forward call. Both kit models accept batched inputs of shape `(B, T)`, returning `logits` of shape `(B, T, V)`. A batched call counts as **one** forward call and $B \cdot T$ tokens processed, and returns a single cache covering the same prefix length for every row. `cache.truncate(n)` trims all rows to `n` tokens, and `cache.select(rows)` keeps only the listed row indices in that order. Because the models are causal, right-padding a shorter row with any token id does not change the logits at its real positions.

**Cost.**

- **Target `forward_calls`** $\le \max_b R_b$. Every round makes one batched target call, and no row waits for another. Truncating all rows to the shortest acceptance in each round would require extra rounds, and decoding rows sequentially would take $\sum_b R_b$ calls.
- **Target `tokens_processed`** $\le 1.25\,T_{\text{ref}}$, where $T_{\text{ref}}$ is the token count of the following reference procedure. Let $c$ be the common cached prefix length valid for all active rows ($c = 0$ initially). In each round, active row $b$ with $\text{len}_b$ verified tokens feeds $\text{len}_b - c + m_b$ tokens (its verified tokens from position $c$ onward, followed by its $m_b$ drafts). The batched call is padded to the widest active row and costs $(\text{active rows}) \times (\text{width})$. After verification, row $b$'s cache entries are valid through $\text{len}_b + n_b$ tokens. Completed rows leave the batch, and $c$ becomes the minimum valid length among the remaining active rows.
- **Draft `tokens_processed`** $\le \sum_b (P + N + D_b)$, where $D_b$ is row $b$'s total drafted tokens (the stage 1 limit summed over all rows). You may call the draft model per row.

### Example 5

Consider `draft_target_pair(fw, seed=4, agreement=0.5)` with `k=4` and 40 new tokens for prompts `[0, 23, 13, 42, 38, 9]` and `[0, 57, 42, 51, 25, 13]`. Decoded alone, these rows need 21 and 19 rounds. In their first eight rounds, they accept `0, 1, 0, 0, 0, 0, 4, 1` and `1, 4, 4, 0, 0, 3, 0, 2` drafts.

The batch must finish in **21** target calls. The reference target processes 405 tokens (limit 506). Truncating both rows to the smaller acceptance each round produces the same tokens in 31 calls, and decoding the rows sequentially takes 40 calls.

## Constraints

- $1 \le P$, $1 \le k \le 8$ at construction, $1 \le \text{max\_new\_tokens}$, and $P + \text{max\_new\_tokens} \le 128$ (the context length of both models). `min_ngram` $\ge 1$, `max_ngram` $\le 8$, and $1 \le B \le 4$.
- **Greedy runs (stages 1, 3, 4, 5).** Emitted tokens are compared exactly against the target's greedy decode, and `rounds` is compared exactly against the procedure. Whenever argmax is taken, the top two logits differ by at least `1e-3`, so minor floating-point differences from call splitting (around `1e-5`) do not change the choice.
- **Exact sampling runs (stages 2 to 4).** Supplied uniforms stay away from decision boundaries: $|u \cdot q_j[x_j] - p_j[x_j]| \ge 10^{-3} \max(u \cdot q_j[x_j],\ p_j[x_j])$ for every checked draft, and $|u \cdot c_{V-1} - c_v| \ge 10^{-4}\, c_{V-1}$ for every inverse-CDF sample and every $v$. Tokens and `rounds` are compared exactly.
- **Statistical sampling cases (stages 2 and 4).** On a pair of 8-token models, the test makes 600 calls with independent uniforms, generating 3 tokens per call. It checks the marginal distribution of each of the 3 positions and the joint distribution of the first 2 positions against the target's exact distribution using finite-sample Chernoff bounds. A correct implementation fails with probability below $10^{-9}$, and seeds are fixed.
- **Cost limits.** The exact counter limits of stages 1 and 5 are checked on every applicable case after verifying the tokens.
- Do not draw your own random numbers. `rounds` may contain tuples or lists.
- The time limit per case is 30 seconds. Because statistical cases invoke `generate` 600 times, keep per-round tensor operations minimal.

## Why this matters

Production inference engines such as llama.cpp, vLLM, and Hugging Face `transformers` rely on this exact loop to combine draft models, n-gram prompt lookup, adaptive speculation, and batched verification. The single-step acceptance formula is only a few lines, while the bugs live in the surrounding loop.

Each component can fail silently. A stale target cache generates fluent text that no longer matches the target model. Sampling from an unnormalized residual, or trusting copied tokens without verification, biases the output distribution toward the proposer in ways that single responses hide. Truncating a batch to its unluckiest row still produces valid tokens, but loses most of the promised speedup.
