Keep two tensors of shape `(N, E)` apart from the first line on: the scores, and the
scores plus the bias. Which of the two decides *who* is chosen, and which of the two
do the weights of the chosen experts come from? In the worked example, expert 2 is
chosen by a biased score of 0.8. What is its weight computed from?

---

- Scores: `sigmoid(x @ w_router.T)`, or the square root of your framework's
  `softplus` of the same logits. There is no softmax anywhere in this problem.
- Selection: a **stable** descending sort of `scores + bias` along the expert axis,
  keeping the first `k` indices. Sorting the negated values with a stable sort ranks
  equal entries by index, lowest first (`torch.argsort(..., stable=True)`;
  `jnp.argsort` and `mx.argsort` are stable).
- Weights: gather the **unbiased** `scores` at those indices
  (`gather` / `take_along_axis`), divide by their sum plus `1e-20`, and only then
  multiply by `scale`.

---

```
logits  = x @ w_router.T
scores  = sigmoid(logits)                       # or sqrt(softplus(logits))
indices = argsort(-(scores + bias), stable)[:, :k]
chosen  = take_along_axis(scores, indices, -1)
weights = chosen / (chosen.sum(-1, keepdims=True) + 1e-20) * scale
```

For the layer, turn `(indices, weights)` into an `(N, E)` matrix that is zero where a
token did not choose an expert, for example
`((indices[..., None] == arange(E)) * weights[..., None]).sum(1)`. Then
`y = swiglu(x, *shared_w)`, and for each expert `e` add
`dense[:, e:e+1] * swiglu(x, w_gate[e], w_up[e], w_down[e])`. The shared expert is
added as it is: no weight, no scale.
