## The idea

The OpenAI definition is one line. For token $j$ with logit $\mu_j$ that has been
sampled $c_j$ times so far:

$$
\mu_j \leftarrow \mu_j - c_j \cdot \alpha_\text{frequency}
- \mathbb{1}[c_j > 0] \cdot \alpha_\text{presence}.
$$

Subtracting $d$ from a logit divides the token's odds against every other token by
$e^{d}$, whatever the logit was. That makes these penalties behave uniformly, unlike
the multiplicative repetition penalty, whose effect depends on the sign and size of
the logit.

- The **frequency** term is linear in the count. Each repetition costs the same
  additional amount, so a token stuck in a loop is pushed down until something else
  wins.
- The **presence** term is a step: it is paid once, when the count goes from 0 to 1,
  and never again. It does not punish using a word many times; it rewards using words
  that have not appeared.

The count is over tokens the model *sampled*. The prompt is input, not output, so it
is excluded. Per row that is a histogram of the generated part of the sequence, and a
histogram over a batch is a scatter-add: every position adds a weight to the column
of its token, and the weight is 1 for generated positions and 0 for prompt positions.

## Reference solution (PyTorch)

```python
def count_generated(tokens, prompt_lengths, vocab_size):
    generated = torch.arange(tokens.shape[1]) >= prompt_lengths[:, None]
    counts = torch.zeros(tokens.shape[0], vocab_size, dtype=torch.int64)
    return counts.scatter_add(1, tokens, generated.to(torch.int64))


def apply_penalties(logits, token_counts, frequency_penalty, presence_penalty):
    penalty = token_counts * frequency_penalty + (token_counts > 0) * presence_penalty
    return logits - penalty
```

- `arange(T) >= prompt_lengths[:, None]` broadcasts to `(B, T)`; each row gets its own
  boundary. Position `prompt_lengths[b]` is the first generated one, hence `>=`.
- Using the mask as the scattered *value* avoids selecting positions, which would
  give a ragged result. Prompt positions still write, but they add 0.
- `scatter_add` (and `.at[rows, tokens].add(...)` in JAX and MLX) accumulates when an
  index repeats. Plain indexed assignment does not.
- An integer tensor times a Python float is float32 in all three frameworks, and so
  is a boolean tensor times a float, so the result needs no cast. A masked logit stays
  masked because `-inf` minus a finite number is `-inf`.

## Common mistakes

- **The multiplicative form.** Dividing positive logits and multiplying negative ones
  by a factor is the *repetition* penalty of CTRL, a different sampler with a
  different parameter range (around 1.0 to 1.3). The names are close enough that the
  two are regularly wired to each other's code path.
- **Presence scaled by the count**, `counts * (frequency + presence)`. The two
  parameters then do the same thing, and a presence penalty of 1 on a token seen ten
  times removes 10 from its logit instead of 1.
- **Counting the prompt.** It is the natural thing to do when the function is handed
  the whole context. With a long prompt the penalties then act mostly on the prompt's
  vocabulary before the model has produced a single token.
- **Off by one at the boundary** (`>` for `>=`): the first generated token of every
  row escapes the penalty.
- **Repeats counted once.** In PyTorch, `counts[rows, tokens] += 1` evaluates the
  right-hand side once and writes each distinct position once, so every count is 0 or
  1. The frequency penalty then silently behaves like a second presence penalty.
- **One prompt length for the batch**, left over from single-sequence code.
- **Adding instead of subtracting.** A positive penalty must make a repeated token
  less likely.

## Complexity

`count_generated` is $O(B \cdot (T + V))$: one pass over the tokens and the allocation
of the table. `apply_penalties` is $O(B \cdot V)$. In a decoding loop the count is not
recomputed from scratch: the engine keeps the `(B, V)` table and adds 1 at the token
it has just sampled.

## In the wild

The formula is quoted from OpenAI's API documentation ("Frequency and presence
penalties"), which defines `c[j]` as how often the token was sampled prior to the
current position, suggests coefficients of about 0.1 to 1 to reduce repetition, and
allows values from -2 to 2. vLLM's `apply_penalties` follows it literally: it builds
bin counts and masks with `scatter_add_` separately for the prompt tokens and the
output tokens, subtracts `frequency_penalties * output_bin_counts` and
`presence_penalties * output_mask`, and uses the prompt mask only for the repetition
penalty, which does cover the prompt. llama.cpp applies the same expression,
`logit -= count * penalty_freq + (count > 0) * penalty_present`, to the counts of a
sliding window of the last `penalty_last_n` tokens (64 by default), and mlx-lm limits
its presence and frequency penalties to the last 20 tokens. A window is a third
answer to "which tokens count", so the same parameter values do not transfer
between engines unchanged.
