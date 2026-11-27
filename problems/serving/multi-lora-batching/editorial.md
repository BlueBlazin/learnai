## The idea

In `lora-merge`, we saw two equivalent ways to evaluate an adapted linear layer: the merged form $x\,(W + \tfrac{\alpha}{r} B A)^{\top}$ and the unmerged form $x W^{\top} + \tfrac{\alpha}{r} (x A^{\top}) B^{\top}$. For a single adapter, merging is cheaper. For a batch that mixes adapters, only the unmerged form works because it splits the computation into two parts:

- A shared term $x W^{\top}$ that is identical for every row regardless of its adapter, costing $\text{in} \times \text{out}$ multiply-adds per token.
- A per-row adapter term that costs only $r\,(\text{in} + \text{out})$ multiply-adds per token.

Compute the first term with a single call to `base(x)` over the entire batch. To compute the second term without a Python loop over rows, gather each row's adapter matrices into `a` and `b` along the batch axis by indexing `a_stack` and `b_stack` with `adapter_ids`. Batched matrix multiplication then multiplies row $b$ by `a[b]` and `b[b]` in parallel.

Two details make this gather exact:

- **Mixed ranks.** Stacking adapters into one tensor requires zero-padding every adapter to $r_{\max}$. The extra zeros do not affect the matrix products, because padded rows of $A_n$ produce zeros that multiply zero columns of $B_n$. However, LoRA scales each adapter by $\alpha / r_n$ using its true rank, so you must gather each row's scale from `ranks`.
- **Base-only rows (`-1`).** Negative indices are valid in PyTorch, JAX, and MLX and index from the end of the tensor, so `a_stack[adapter_ids]` would silently assign the last adapter to rows with ID `-1`. The branch-free fix is to clamp negative IDs to `0` in an `index` tensor (any valid index works) and zero out their scale with `where(adapter_ids >= 0, alpha / ranks[index], 0.0)`. The base output for those rows remains untouched.

## Reference solution (PyTorch)

```python
def batched_lora_forward(x, base, a_stack, b_stack, ranks, adapter_ids, alpha):
    y = base(x)  # once, whole batch
    has_adapter = adapter_ids >= 0
    index = torch.clamp(adapter_ids, min=0)
    a = a_stack[index]  # (B, r_max, in)
    b = b_stack[index]  # (B, out, r_max)
    scale = torch.where(has_adapter, alpha / ranks[index], 0.0)  # (B,)
    tokens = x if x.dim() == 3 else x[:, None, :]  # (B, T, in)
    low = tokens @ a.transpose(1, 2)  # (B, T, r_max)
    delta = (low @ b.transpose(1, 2)) * scale[:, None, None]
    return y + delta.reshape(y.shape)
```

The JAX and MLX solutions follow the same structure, using `maximum(adapter_ids, 0)`, `where`, and `swapaxes`.

## Common mistakes

- **Looping over rows for the full computation.** Calling `base` inside a loop over batch rows turns one shared matmul into $B$ small ones and $2B$ adapter matmuls. This fails the cost test (6 base calls instead of 1) and runs 16x to over 100x slower on the speed benchmark.
- **Looping over rows for the adapter updates only.** This makes one `base` call and passes the cost test, but still pays per-row Python overhead and fails the speed test.
- **Splitting the batch by adapter and calling `base` per group.** Grouping rows by adapter and calling `base` on each group splits the shared weight matmul into one call per active adapter, failing the cost test. (Grouping by adapter for the *updates* only, after a single `base(x)` call, is a valid alternative to gathering.)
- **Indexing stacks directly with `-1`.** `a_stack[adapter_ids]` succeeds without error and silently applies the last adapter in the stack to base-model requests.
- **Masking the full output instead of the update.** Computing `(y + delta) * mask` zeroes out the base-model rows instead of leaving `y` intact.
- **Scaling by the padded rank.** Using `alpha / a_stack.shape[1]` is right only for adapters with rank $r_{\max}$. A rank-1 adapter in a stack padded to 8 is scaled down to one-eighth of its strength.
- **Applying a single adapter to the whole batch.** Reading only `adapter_ids[0]` and applying that adapter to every row breaks mixed batches.
- **Broadcasting `scale` on the wrong axis.** For 3D `x` of shape `(B, T, in)`, a `scale` of shape `(B, 1)` aligns with `(T, out)` instead of `(B, 1, 1)`. This raises a shape error when $B \ne T$ and silently computes wrong values when $B = T$.

## Complexity

The single `base(x)` call takes $O(B\,T \cdot \text{in} \cdot \text{out})$ operations. The gathered adapter updates take $O(B\,T\,r_{\max}\,(\text{in} + \text{out}))$ operations.

Gathering copies $B\,r_{\max}\,(\text{in} + \text{out})$ adapter weights into memory so each row has its own padded slice. Production systems avoid this copy and padding overhead either by grouping rows by adapter (one unpadded matmul per active adapter) or by using custom kernels that read adapter weights indirectly through `adapter_ids`.

## In the wild

- **Punica** ("Punica: Multi-Tenant LoRA Serving", arXiv 2310.18547) introduced Segmented Gather Matrix-Vector multiplication (SGMV), a CUDA kernel that computes the $x A^{\top} B^{\top}$ updates for mixed-adapter batches by grouping requests that share an adapter while a single base backbone serves all of them.
- **S-LoRA** ("S-LoRA: Serving Thousands of Concurrent LoRA Adapters", arXiv 2311.03285) uses the same split between a batched base forward pass and separate adapter updates. It avoids weight merging for multi-adapter serving, adds custom kernels for heterogeneous ranks without zero-padding, and manages adapter weights and KV cache in a unified memory pool.
- **vLLM** serves per-request adapters (`--lora-modules name=path`, selected via the request's `model` field) and mixes base-model and adapter requests in the same batch. In `vllm/lora/punica_wrapper`, adapters live in stacked tensors indexed per token by `indices`, where `add_lora_linear` computes `y[i] += x[i] @ lora_a_stacked[indices[i]] @ lora_b_stacked[indices[i]] * scale` and `-1` means no adapter. Because vLLM flattens tokens into one batch axis, its layout matches the 2D `(B, in)` case with hundreds of rows.
- **Hugging Face PEFT** supports mixed-adapter inference batches through the `adapter_names` argument to `forward`, taking one adapter name per batch row and a reserved name for base-model rows.
