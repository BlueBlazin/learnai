Implement the two ends of a decoder forward pass (token embedding lookup and the final logit projection) along with three model-specific scaling constants. These constants are an embedding multiplier, a logit divisor, and an optional logit soft cap.

![This problem is the token embedding and the final linear layer, each with a constant from the config.](assets/map.svg)

## Specification

```python
def embed(tokens, table, scale): ...
def finalize_logits(h, w_out, logit_scale, softcap=None): ...
```

Let `V` be the vocabulary size and `d` be the model's hidden dimension.

**`embed(tokens, table, scale)`** indexes into the embedding matrix $E$ (`table`, with shape `(V, d)`) using integer token IDs in $[0, V)$, then multiplies the looked-up vectors by `scale`:

$$
x = \texttt{scale} \cdot E[\texttt{tokens}] .
$$

`embed` must not modify `table` in place. In a model with tied embeddings, the output head shares the same matrix `table`, and the multiplier applies only to the input embeddings, not to the output head.

**`finalize_logits(h, w_out, logit_scale, softcap=None)`** converts final hidden states `h` into vocabulary logits in three steps, in order:

1. **Project.** Compute $z = h\,W_{\text{out}}^\top$, where `w_out` has shape `(V, d)` in `(out, in)` layout and no bias.
2. **Divide.** Rescale the logits by `logit_scale`: $z \leftarrow z \,/\, \texttt{logit\_scale}$. Note that `logit_scale` is a divisor, not a multiplier.
3. **Soft-cap.** If `softcap` is not `None`, bound the divided logits smoothly with $c = \texttt{softcap}$:

$$
z \leftarrow c \cdot \tanh\!\left(\frac{z}{c}\right) .
$$

The soft cap is smooth and strictly increasing. A logit whose magnitude is much smaller than $c$ barely changes, while large logits saturate near $\pm c$ without exceeding it.

## Shapes

| name | shape | dtype |
|---|---|---|
| `tokens` | `(...)`, zero to two dimensions | integer in $[0, V)$ |
| `table` | `(V, d)` | float32 |
| `scale` | Python float | |
| result of `embed` | `(..., d)` | float32 |
| `h` | `(..., d)`, one to three dimensions | float32 |
| `w_out` | `(V, d)` | float32 |
| `logit_scale` | Python float ($> 0$) | |
| `softcap` | `None` or Python float ($> 0$) | |
| result of `finalize_logits` | `(..., V)` | float32 |

## Example

```python
table = [[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]]  # V = 3, d = 2

embed([2, 0], table, 12.0)  # [[6.0, -12.0], [12.0, 24.0]]

h = [3.0, 1.0]
finalize_logits(h, table, 4.0)  # [1.25, 3.25, 0.125]
finalize_logits(h, table, 4.0, 2.0)  # [1.109199, 1.850692, 0.124837]
```

Here `table` is reused as `w_out` (a tied head). Projecting `h` gives $z = [3 + 2,\; 9 + 4,\; 1.5 - 1] = [5, 13, 0.5]$, and dividing by 4 yields $[1.25, 3.25, 0.125]$. Applying a soft cap of $c = 2$ computes $2\tanh(1.25 / 2) = 1.1092$, $2\tanh(3.25 / 2) = 1.8507$, and $2\tanh(0.125 / 2) = 0.1248$. The logit of $3.25$ is pulled just under 2, while $0.125$ changes by about a tenth of a percent.

Here are the constants used by three real model families:

| model | `scale` | `logit_scale` | `softcap` |
|---|---|---|---|
| Llama 3 | 1.0 | 1.0 | `None` |
| Granite 4.0 350M | 12.0 (`embedding_multiplier`) | 4.0 (`logits_scaling`) | `None` |
| Gemma 2 | $\sqrt{d}$ | 1.0 | 30.0 (`final_logit_softcapping`) |

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. A NaN is always wrong.
- $1 \le V, d \le 64$. `tokens` has at most 64 elements and `h` has at most 1,024 elements.
- $|\texttt{scale}| \le 100$, $0.1 \le \texttt{logit\_scale} \le 100$, and $0.1 \le \texttt{softcap} \le 100$.
- Logits before the soft cap can be up to a thousand times larger than `softcap`, and the output must still be finite.
- No Python loops over tokens or positions are needed.

## Why this matters

When a ported model almost works, these constants are often the culprit. They live in `config.json` rather than the checkpoint tensors and never affect tensor shapes, so omitting one raises no error.

Without Granite's embedding multiplier, the residual stream starts twelve times too small and every normalization layer behaves differently than during training. Without its logit divisor, logits are four times too large. Greedy decoding hides this bug because the argmax is unchanged, while sampling behaves as if the temperature were 0.25, producing repetitive, overconfident text that looks like a sampler bug. Gemma's soft cap bounds logits at 30, reshaping probabilities on the tokens the model is most confident about. Finally, folding `scale` into the embedding table when loading weights silently corrupts the output head of every model with tied embeddings. `real-model-logits` tests these constants against a real checkpoint, while this problem isolates the arithmetic.
