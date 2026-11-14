Until now every model in these problems was a toy with random weights. This one is real: **Granite 4.0 350M**, a 352-million-parameter decoder. Implement the last step of its forward pass.

Write `next_token_logits(hidden, norm_weight, unembed)`. `hidden` holds the final hidden states of a prompt, shape $(T, d)$. Apply the final RMS normalization to the **last** position only, then project it onto the vocabulary:

$$
\hat{h} = \frac{h_T}{\sqrt{\tfrac{1}{d}\sum_i h_{T,i}^2 + \varepsilon}} \odot g,
\qquad
z = W_U \, \hat{h}
$$

with $\varepsilon = 10^{-5}$, gain $g$ = `norm_weight` of shape $(d,)$, and $W_U$ = `unembed` of shape $(V, d)$.

## Why only the last position

Greedy decoding needs one row of logits per step, not $T$ of them. Normalizing and projecting every position costs $T \cdot V \cdot d$ multiplications; the last position alone costs $V \cdot d$.

| | positions projected | multiplications at $T = 512$ |
|---|---|---|
| project everything | $T$ | $512 \cdot V \cdot d$ |
| last position only | $1$ | $V \cdot d$ |

## Cases

The first cases use small random tensors. The rest read `norm_weight` and `unembed` from the downloaded checkpoint and compare against logits computed by the reference implementation, so they run only once the model is on disk.

```python
def next_token_logits(hidden, norm_weight, unembed):
    """Return logits of shape (V,) for the last position."""
```
