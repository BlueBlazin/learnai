## The idea

In the OpenAI API specification, a token $j$ with logit $\mu_j$ that has been sampled
$c_j$ times so far is adjusted by:

$$
\mu_j \leftarrow \mu_j - c_j \cdot \alpha_\text{frequency}
- \mathbb{1}[c_j > 0] \cdot \alpha_\text{presence}.
$$

Subtracting $d$ from a logit divides that token's odds against every other token by
$e^{d}$, regardless of the logit's original value. This gives additive penalties a
uniform effect, unlike the multiplicative repetition penalty, whose strength depends on
the sign and magnitude of the logit.

- The **frequency** term grows linearly with the count. Each extra repetition incurs the
  same additional cost, pushing a looping token down until another token wins.
- The **presence** term is a step function. It is paid once when the count moves from 0
  to 1 and never again. Rather than punishing repeated use of a word, it encourages the
  model to pick words that have not appeared yet.

The count covers only tokens the model sampled. Prompt tokens are input, not output, so
they are excluded. Per row, this count is a histogram of the generated suffix. Across a
batch, you can compute it with a scatter-add where every position adds a weight to its
token's column, using weight `1` for generated positions and weight `0` for prompt
positions.

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

- `arange(T) >= prompt_lengths[:, None]` broadcasts to shape `(B, T)` so each row uses
  its own boundary. Position `prompt_lengths[b]` is the first generated position, which
  is why the comparison is `>=`.
- Passing the mask as the scattered values avoids indexing out ragged slices of
  generated tokens. Prompt positions still participate in the scatter, but they add `0`.
- `scatter_add` (and `.at[rows, tokens].add(...)` in JAX and MLX) accumulates when an
  index repeats. Plain indexed assignment overwrites instead.
- Multiplying an integer or boolean tensor by a Python float produces a float32 tensor
  in all three frameworks, so no explicit cast is needed. Masked logits remain `-inf`
  because `-inf` minus any finite number is `-inf`.

## Common mistakes

- **Using the multiplicative form.** Dividing positive logits and multiplying negative
  logits by a factor is the CTRL repetition penalty, which is a different sampler with a
  different parameter range (around `1.0` to `1.3`). Because the names are similar, the
  two are often wired to the wrong formula.
- **Scaling the presence penalty by the count.** Computing
  `counts * (frequency + presence)` makes both parameters do the same thing. A presence
  penalty of `1.0` on a token seen ten times then subtracts `10.0` instead of `1.0`.
- **Counting prompt tokens.** When a function receives the full context, it is easy to
  count every position. With a long prompt, the penalties then suppress the prompt's
  vocabulary before the model has generated a single token.
- **Off-by-one boundary (`>` instead of `>=`).** Using strict inequality skips the first
  generated token of every row.
- **Counting repeats only once.** In PyTorch, `counts[rows, tokens] += 1` evaluates the
  right-hand side once and writes each duplicate index only once, capping every count at
  `1`. The frequency penalty then silently behaves like a second presence penalty.
- **Using one prompt length for the whole batch.** This bug often carries over from
  single-sequence code.
- **Adding instead of subtracting.** A positive penalty must lower the logit of a
  repeated token, not raise it.

## Complexity

`count_generated` runs in $O(B \cdot (T + V))$ time for one pass over the tokens and
allocating the `(B, V)` table. `apply_penalties` runs in $O(B \cdot V)$ time. In a real
decoding loop, the engine does not recompute counts from scratch at every step. It keeps
the `(B, V)` table in memory and increments the entry for each newly sampled token.

## In the wild

The formula comes from OpenAI's API documentation ("Frequency and presence penalties"),
which defines $c_j$ as how often token $j$ was sampled prior to the current step,
recommends values around `0.1` to `1.0` to reduce repetition, and accepts values in
$[-2, 2]$.

vLLM's `apply_penalties` follows this definition directly. It uses `scatter_add_` to
build separate bin counts and masks for prompt tokens and output tokens, subtracts
`frequency_penalties * output_bin_counts` and `presence_penalties * output_mask`, and
uses the prompt mask only for the repetition penalty (which does cover the prompt).

Other engines scope the count differently. llama.cpp applies the same formula,
`logit -= count * penalty_freq + (count > 0) * penalty_present`, over a sliding window
of the last `penalty_last_n` tokens (64 by default). mlx-lm restricts its presence and
frequency penalties to the last 20 tokens. Because the counting window differs, the same
penalty values do not transfer unchanged across engines.
