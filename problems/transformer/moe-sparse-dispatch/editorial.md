## The idea

The dense MoE layer in `moe-router-forward` evaluates all $E$ experts on every token and multiplies $E - k$ of the results by zero. The sparse layer computes only the $k$ active terms per token. It evaluates the same sum,

$$
y_t = \sum_{e \in S_t} w_{t,e}\, f_e(x_t),
$$

but reorganizes the work by expert instead of by token. Letting $T_e$ be the set of tokens routed to expert $e$, the output is:

$$
y = \sum_{e\,:\,T_e \ne \emptyset} \operatorname{scatter}_{T_e}\!\big( w_{T_e, e} \odot f_e(x_{T_e}) \big).
$$

Each active expert performs three steps:

1. **Gather** the rows $x_{T_e}$ into a contiguous $(|T_e|, D)$ batch.
2. **Run** expert $e$ once on that batch. This is the only expensive step, and batching turns it into a few large matrix products.
3. **Scatter-add** the weighted outputs back into rows $T_e$ of `y`.

Because every token belongs to exactly $k$ sets, the row counts sum to $\sum_e |T_e| = kN$.

To build the sets $T_e$ without looping over tokens, start from the routing table `chosen` of shape `(N, k)`. Flatten it into $kN$ `(token, expert, weight)` assignments and sort them by expert index. In this sorted order, each expert owns one contiguous run whose length is the number of tokens routed to it. This counting-sort layout relies only on `argsort`, `bincount`, and integer indexing, avoiding boolean-mask reads and data-dependent tensor shapes. Finally, convert the run lengths to Python ints so you can slice each run and skip experts with count 0.

## Reference solution (PyTorch)

```python
def moe_sparse(x, w_router, experts, k):
    n_tokens = x.shape[0]
    logits = x @ w_router.T
    chosen = torch.argsort(-logits, dim=-1, stable=True)[:, :k]
    weights = torch.softmax(torch.gather(logits, 1, chosen), dim=-1)

    flat_expert = chosen.reshape(-1)
    flat_token = torch.arange(n_tokens).repeat_interleave(k)
    flat_weight = weights.reshape(-1)
    order = torch.argsort(flat_expert, stable=True)
    counts = torch.bincount(flat_expert, minlength=experts.n_experts).tolist()

    y = torch.zeros_like(x)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        run = order[start : start + count]
        start += count
        tokens = flat_token[run]
        out = experts(expert, x[tokens])
        y = y.index_add(0, tokens, flat_weight[run, None] * out)
    return y
```

- **Weights.** The renormalized probability $p_e / \sum_{j \in S_t} p_j$ equals the softmax over just the $k$ selected logits, because the full softmax denominator cancels out. No full softmax over all $E$ experts is needed.
- **Alignment.** `tokens`, `flat_weight[run]`, and the rows of `out` all share the same ordering within `run`, so row $j$ of the expert output always pairs with the matching weight and destination token. Any permutation of `run` produces the same result as long as all three arrays use it.
- **Accumulation.** Each row of `y` receives $k$ updates, one from each expert chosen by that token. Within a single expert's run, token indices are distinct (a token never selects the same expert twice), so indexed `+=` is also safe within each step while accumulating across experts.

The JAX and MLX references differ in two lines. They compute `counts` on the host with `np.bincount`, and they scatter-add with `y.at[tokens].add(...)`.

## Common mistakes

- **Dense evaluation.** Running every expert on every token produces the right numbers at $E / k$ times the compute cost (4x for Mixtral, 16x for Qwen3-30B-A3B).
- **One call per token.** Looping `for t in range(N): experts(e, x[t:t+1])` matches the exact row count but makes $kN$ calls. On real hardware, each call triggers a separate kernel launch and a single-row matmul, collapsing throughput.
- **Grouping per top-$k$ slot.** Grouping first-choice assignments separately from second-choice assignments feels natural after calling `topk`, but any expert chosen as first choice by some tokens and second choice by others gets called twice. Flatten all $k$ slots before grouping.
- **Calling idle experts.** Looping `for e in range(E)` without skipping zero-count experts issues empty calls. With 128 experts and small batches, most launches do no work.
- **Scattering to a contiguous slice.** Writing an expert's output to `y[first : first + n]` assumes its tokens are adjacent. That passes toy tests with uniform inputs and fails on real batches.
- **Assigning instead of accumulating.** Writing `y[tokens] = ...` overwrites earlier experts and keeps only the last one. This bug passes when $k = 1$ and breaks for $k > 1$.
- **Forgetting to renormalize.** Gathering raw entries from the full softmax produces weights that sum to less than 1 and shrinks every MoE output.

## Complexity

The router projection takes $O(NDE)$ time, sorting the assignments takes $O(kN \log(kN))$ time over small integers, and the expert MLPs process $kN$ rows instead of $EN$. The number of expert calls is at most $\min(E, kN)$. The gathered batches total $O(kND)$ elements across all groups, processed one group at a time (at most $O(ND)$ peak per call).

## In the wild

Hugging Face `transformers` implements Mixtral's `MixtralSparseMoeBlock.forward` and the Qwen3 MoE block in this style. It takes `topk` of the router softmax, renormalizes the weights, builds a one-hot selection mask, finds `expert_hit` (the active experts), and loops only over those experts. For each active expert, it locates its tokens with `torch.where`, runs the expert on `hidden_states[top_x]`, and accumulates results with `index_add_`.

Fused kernels avoid materializing separate per-expert batches altogether. vLLM's fused MoE calls `moe_align_block_size` to sort token IDs by expert and pad each group to the kernel's block size, then executes a single grouped matmul over the sorted rows. In `mlx-lm`, `SwitchGLU` gathers expert weights per token directly, and whenever a call has 64 or more assignments it sorts them by expert index first and unsorts afterward for memory locality. Both designs sort for the same reason as this reference: contiguous runs per expert. That same grouping is also the first step of expert parallelism, where each contiguous run is sent to the device holding its expert.
