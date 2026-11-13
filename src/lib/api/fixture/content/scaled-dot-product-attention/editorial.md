The implementation is four lines; the lesson is in the order of operations.

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
return weights @ v
```

## Why mask before the softmax

Zeroing weights after the softmax leaves rows that no longer sum to $1$: the model would see attenuated values at every masked position. Setting scores to $-\infty$ first removes those keys from the normalizer, which is what "cannot attend" means.

## Why $\sqrt{d}$

If the components of $q$ and $k$ are independent with unit variance, $q \cdot k$ has variance $d$. Dividing by $\sqrt{d}$ returns it to $1$. Without the scale, a head with $d = 128$ would see score gaps about eleven times larger, and softmax would collapse onto a single key.
