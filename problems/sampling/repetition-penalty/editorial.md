## The idea

The penalty depends on one question per (row, token) pair: is this token in this
row's history? That is a boolean table `seen` of shape `(B, V)`, and the history
`prev_tokens` is a list of column indices into it, one list per row. Turning lists of
indices into a table is a **scatter**: start from all `False` and write into
`seen[b, prev_tokens[b, t]]` for every `b` and `t` in one operation.

Scatter handles the two awkward requirements without extra code:

- **Duplicates.** A token that occurs several times is written several times. Since
  every write stores the same value, the result is the same as writing once. Set
  membership falls out of the operation, with no `unique` and no counting.
- **Padding.** Instead of filtering padding out (which would make rows ragged), choose
  the value written: `prev_tokens != pad_id`. A padding entry writes `False` into the
  pad column, which was `False` already. All writes to one position agree, so the order
  in which the framework applies them does not matter.

With `seen` in hand, compute the penalized value for every logit from the original
logits and select.

## Reference solution (PyTorch)

```python
def apply_repetition_penalty(logits, prev_tokens, penalty, pad_id):
    seen = torch.zeros_like(logits, dtype=torch.bool).scatter(1, prev_tokens, prev_tokens != pad_id)
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalized, logits)
```

In JAX the scatter is `jnp.zeros(shape, bool).at[rows, prev_tokens].set(...)` with
`rows = jnp.arange(B)[:, None]`; in MLX it is `mx.put_along_axis(..., axis=1)`.

An equally good formulation works on the history side instead of the vocabulary side:
gather the logits at `prev_tokens`, penalize those `(B, T)` values, restore the ones
at padding positions, and scatter them back into a copy of the logits. It touches
only $B \cdot T$ entries plus one copy, and it is correct for the same reason:
duplicate positions all receive the same once-penalized value, because every value is
computed from the original logits.

## Why the sign matters

Dividing a positive logit by $\theta > 1$ lowers it. Dividing a negative logit by
$\theta$ moves it toward zero, which *raises* it: $-1.0 / 1.25 = -0.8$. The penalty
must move every seen logit in the direction of lower probability, so negative logits
are multiplied. A logit of exactly `0` is unchanged either way, and `-inf` stays
`-inf` under multiplication.

## Common mistakes

- **Always dividing.** This is the formula as printed in the CTRL paper. Division
  lowers a score only if the score is positive; on logits, which are negative about
  half of the time, it rewards repeated tokens whose logits are negative. Swapping
  the two branches rewards every repeated token.
- **Penalizing padding.** Scattering `True` for every entry marks the pad column in
  every row that has padding. The pad id is a real vocabulary entry, frequently the
  end-of-sequence token, so short sequences in a batch get a suppressed EOS.
- **Penalizing per occurrence.** Counting occurrences (`scatter_add`, or a loop that
  updates the logits in place token by token) applies $\theta^{\text{count}}$. The
  definition is about set membership. In-place loops fall into this by accident: the
  second occurrence penalizes an already penalized value.
- **One seen set for the batch.** Indexing columns with the flattened history
  (`logits[:, prev_tokens.flatten()]`) penalizes every row for what any row generated.
- **A Python loop over rows or tokens.** Correct, and one to two orders of magnitude
  slower: each iteration pays tens of microseconds of interpreter and dispatch
  overhead to move a few floats. The speed case measures about 100x for a loop over
  rows and more for a loop over tokens.

## Complexity

$O(B \cdot (T + V))$ time and $O(B \cdot V)$ extra memory for the mask and the result.
A one-hot comparison `prev_tokens[:, :, None] == arange(V)` is also loop-free but
builds a `(B, T, V)` intermediate, $O(B \cdot T \cdot V)$ time and memory; it works at
small sizes and stops being reasonable at real vocabulary sizes ($V \approx 10^5$).

## In the wild

The penalty comes from Keskar et al., "CTRL: A Conditional Transformer Language Model
for Controllable Generation" (2019), which samples from
$p_i \propto \exp\!\big(x_i / (T \cdot I(i \in g))\big)$ with $I = \theta$ for
previously generated tokens and suggests $\theta \approx 1.2$. Hugging Face's
`RepetitionPenaltyLogitsProcessor` uses the sign-aware rule of this problem with the
gather, `where`, scatter formulation; llama.cpp's `repeat_penalty` and vLLM's
`repetition_penalty` apply the same rule. Engines differ in what counts as history
(Hugging Face includes the prompt, llama.cpp a window of the last `repeat_last_n`
tokens), which is why the same penalty value behaves differently across them. The
frequency and presence penalties of the OpenAI-style APIs are the additive relatives:
they subtract from the logit, and the frequency penalty deliberately does scale with
the number of occurrences.
