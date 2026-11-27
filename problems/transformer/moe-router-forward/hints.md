For one token, the router produces $E$ logits. Which axis of the logits tensor `(B, T, E)` do softmax and top-k act on? What shape must the final weights have so that weight `[b, t, e]` multiplies the output of expert `e` for token `[b, t]`? If you wrote one token's weights as a vector of length $E$, what would it look like for $k = 2$?

---

Aim for a dense weight tensor `w` of shape `(B, T, E)` where each token's `k` selected experts hold their renormalized probabilities and the remaining `E - k` entries are 0. Once you have `w`, you can evaluate every expert on all of `x` and sum `w[..., e:e+1] * expert_e(x)` over `e`.

To build `w`, compute `probs = softmax(logits)` along the last axis, construct a boolean mask `keep` of the same shape marking each token's top `k` experts, zero out the unselected entries, and divide by the sum along the last axis (using `keepdim`).

Building `keep` requires ranking every expert within each token, breaking ties in favor of the lowest index. A framework's `topk` does not guarantee tie order, whereas a stable sort does.

---

To rank experts along the last axis by decreasing logit with ties going to the lowest index, sort the negated logits with a stable `argsort`, then invert the permutation with a second `argsort`:

```python
order = torch.argsort(-logits, dim=-1, stable=True)  # order[..., r] = expert with rank r
rank = torch.argsort(order, dim=-1)                  # rank[..., e] = rank of expert e
weights = torch.where(rank < k, probs, 0.0)
weights = weights / weights.sum(dim=-1, keepdim=True)
```

`jnp.argsort` and `mx.argsort` are already stable by default. Then loop over the experts and accumulate `y = y + weights[..., e : e + 1] * ops.swiglu(x, w_gate[e], w_up[e], w_down[e])`.
