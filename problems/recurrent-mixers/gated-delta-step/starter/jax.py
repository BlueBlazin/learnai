import jax
import jax.numpy as jnp


def gdn_gates(
    a: jax.Array, b: jax.Array, A_log: jax.Array, dt_bias: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """The decay and the write strength of each head, from their raw projections.

    a, b: float32, shape (T, H). One row per token, one column per head.
    A_log, dt_bias: float32, shape (H,).
    Returns (g, beta), both float32 of shape (T, H):
        g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b).
    """
    raise NotImplementedError


def gdn_step(
    S: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, beta: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """One token of the gated delta rule: decay, write, then read.

    S: float32, shape (H, Dv, Dk). The memory of each head; S @ key is a value.
    q, k: float32, shape (H, Dk). Used as given (not normalized here).
    v: float32, shape (H, Dv).
    g, beta: float32, shape (H,). One row of the gates.
    Returns (y, S): y float32 of shape (H, Dv), read from the updated memory, and
    the updated memory, float32 of shape (H, Dv, Dk).
    """
    raise NotImplementedError
