For one token, the router produces $E$ numbers. Which axis of the logits tensor
`(B, T, E)` do the softmax and the top-k act on, and what shape must the final weights
have so that weight `[b, t, e]` multiplies the output of expert `e` for token `[b, t]`?
If you wrote the weights of one token as a vector of length $E$, what would it look
like for $k = 2$?

---

Aim for a dense weight tensor `w` of shape `(B, T, E)` in which the $k$ selected
experts of each token hold their renormalized probabilities and the other $E - k$
entries are 0. Then the output is just $\sum_e$ `w[..., e:e+1] * expert_e(x)`, with
every expert evaluated on all of `x`.

To build `w`: take `probs = softmax(logits)` over the last axis, compute a boolean
`keep` of the same shape that marks each token's top $k$, zero the rest, and divide by
the sum over the last axis (with `keepdim`).

For `keep` you need the rank of every expert within its token, with ties going to the
lowest index. A framework `topk` does not promise an order among equal values. A
stable sort does.

---

The rank of each entry along the last axis, by decreasing logit, ties to the lowest
index: sort the negated logits with a stable argsort, then invert the permutation with
a second argsort.

```python
order = torch.argsort(-logits, dim=-1, stable=True)  # order[..., r] = expert with rank r
rank = torch.argsort(order, dim=-1)                  # rank[..., e] = rank of expert e
weights = torch.where(rank < k, probs, 0.0)
weights = weights / weights.sum(dim=-1, keepdim=True)
```

(`jnp.argsort` and `mx.argsort` are stable as they are.) Then loop over the experts:
`y = y + weights[..., e : e + 1] * ops.swiglu(x, w_gate[e], w_up[e], w_down[e])`.
