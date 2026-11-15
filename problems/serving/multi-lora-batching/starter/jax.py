import jax
import jax.numpy as jnp


def batched_lora_forward(
    x: jax.Array,
    base,
    a_stack: jax.Array,
    b_stack: jax.Array,
    ranks: jax.Array,
    adapter_ids: jax.Array,
    alpha: float,
) -> jax.Array:
    """One linear layer for a batch whose rows use different LoRA adapters.

    x: float32 (B, in) or (B, T, in); row b belongs to request b.
    base: the frozen base layer, a callable: base(x) is x @ W.T, shape (..., out).
    a_stack: float32 (N, r_max, in), the down-projections, zero-padded to r_max rows.
    b_stack: float32 (N, out, r_max), the up-projections, zero-padded to r_max columns.
    ranks: int (N,), the true rank of each adapter, 1 <= ranks[n] <= r_max.
    adapter_ids: int (B,), the adapter of each row, in [-1, N); -1 means no adapter.
    alpha: Python float; adapter n is scaled by alpha / ranks[n].

    Returns a float32 tensor of the shape of base(x): for a row with adapter n,
    base(x) + (alpha / ranks[n]) * (x @ A_n.T) @ B_n.T; for a row with id -1, base(x).
    """
    raise NotImplementedError
