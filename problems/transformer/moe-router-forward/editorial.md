## The idea

An MoE layer replaces a Transformer block's single MLP with $E$ experts and a router. For each token, the router builds a sparse weight vector $w \in \mathbb{R}^E$ with $k$ nonzero entries that sum to 1, and the layer outputs $\sum_e w_e f_e(x)$. Because $w$ is sparse, only $k$ of the $E$ experts need to run for that token. This lets a model store $E$ MLPs' worth of parameters while paying the compute cost of only $k$.

For each token, the weight vector is built in three steps. First, take the softmax over all $E$ router logits $r_e$ to get probabilities $p_e$. Next, keep the top $k$ experts (the set $S$). Finally, divide the kept probabilities by their sum.

These steps simplify through an algebraic identity. For any selected expert $e \in S$, the full softmax's partition function $Z = \sum_{j=1}^E e^{r_j}$ cancels out:

$$
w_e = \frac{p_e}{\sum_{j \in S} p_j}
    = \frac{e^{r_e} / Z}{\sum_{j \in S} e^{r_j} / Z}
    = \frac{e^{r_e}}{\sum_{j \in S} e^{r_j}} .
$$

Because $Z$ cancels, "softmax over all experts, top-k, then renormalize" computes the exact same function as "top-k of the logits, then softmax over the selected logits." Either order is a valid solution. The two diverge only when implemented incorrectly, such as skipping renormalization in the first form or masking unselected logits with something other than $-\infty$ in the second.

Selection is the one discontinuous step. When a token's $k$-th and $(k+1)$-th logits are equal, the output still needs a deterministic answer, which is why the specification defines a tie-breaking rule. Because `topk` in PyTorch, JAX, and MLX does not guarantee how ties are ordered, the reference solution ranks experts with a stable sort.

## Reference solution (PyTorch)

```python
def moe(x, w_router, experts_w, k):
    w_gate, w_up, w_down = experts_w
    logits = x @ w_router.T  # (B, T, E)
    probs = ops.softmax(logits)
    order = torch.argsort(-logits, dim=-1, stable=True)
    rank = torch.argsort(order, dim=-1)
    weights = torch.where(rank < k, probs, 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    y = torch.zeros_like(x)
    for e in range(w_router.shape[0]):
        y = y + weights[..., e : e + 1] * ops.swiglu(x, w_gate[e], w_up[e], w_down[e])
    return y
```

Sorting `-logits` with `stable=True` lists each token's experts from highest to lowest logit while placing tied experts in ascending index order. A second `argsort` inverts that permutation to produce each expert's rank, so `rank < k` yields a dense selection mask. The top expert always has probability at least $1/E$, so the renormalization denominator is never zero.

Using the kit, the equivalent "top-k then softmax" form is a single line because `ops.top_k_filter` applies the same tie-breaking rule and replaces unselected entries with $-\infty$:

```python
weights = ops.softmax(ops.top_k_filter(logits, k))
```

You could replace the loop over experts with a single `einsum` over the stacked weights. The loop is clearer and matches the structure needed for a sparse implementation.

## Common mistakes

- **No renormalization.** Using raw top-k softmax probabilities produces weights that sum to less than 1 by a different amount for each token. A token with an uncertain router gets a smaller MLP update than a confident one. If all experts are identical, the layer should match that single expert's output, but without renormalization it is scaled down.
- **Masking with 0 instead of $-\infty$.** In the "top-k then softmax" form, setting an unselected logit to 0 still gives it weight $e^0 = 1$ in the softmax numerator. When selected logits are negative, unselected experts end up dominating the sum. Only $-\infty$ zeroes out an entry in softmax.
- **Routing per sequence.** Pooling across the sequence (or using only the first token) gives weights of shape `(B, 1, E)` that silently broadcast across `T`. Routing must happen per token because neighboring tokens routinely pick different experts.
- **Softmax on the wrong axis.** Logits have shape `(B, T, E)`. Applying softmax over axis 1 normalizes each expert across tokens instead of normalizing across experts for each token. When $T = E$, tensor shapes will not catch this bug.
- **Uniform weights.** Averaging the $k$ selected experts ignores how strongly the router preferred each one. This happens to pass when $k = 1$ and fails for $k > 1$.
- **Breaking ties toward the highest index.** Sorting in ascending order and taking the last $k$ entries picks the highest index on ties. Exact ties are rare with trained weights, but zeroed or duplicated router rows trigger them, and relying on an unspecified sort order breaks reproducibility.

## Complexity

Dense evaluation costs $O(B\,T\,E\,D\,H)$ for the experts, $O(B\,T\,E\,D)$ for the router, and $O(B\,T\,E \log E)$ for sorting. That performs $E / k$ times more expert arithmetic than necessary. A sparse implementation runs each expert only on the tokens that selected it, reducing the expert cost to $O(B\,T\,k\,D\,H)$, the same as running $k$ standard MLPs.

## In the wild

Hugging Face's `MixtralSparseMoeBlock` follows this exact sequence. It applies a bias-free linear `gate`, runs `F.softmax` over the experts, calls `torch.topk` with `top_k = 2`, divides the kept weights by their sum, and loops over experts to run each on its assigned tokens before scattering results back with `index_add_`. Mixtral's experts are also SwiGLU MLPs, where `w1`, `w3`, and `w2` correspond to gate, up, and down.

Routing rules vary across model families. DeepSeek's MoE models add shared experts that run on every token and score routed experts differently, and some architectures skip renormalization altogether. As with norm placement and RoPE layout, always verify against the reference implementation of the model you are porting.

In production serving, MoE changes system bottlenecks more than math. All $E$ experts must stay resident in memory even though each token touches only $k$ of them, making memory capacity and bandwidth the binding constraints. Within a batch, each expert also receives a different, unpredictable number of tokens, creating ragged per-expert batches.
