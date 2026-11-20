import jax
import jax.numpy as jnp

from learnai.kit import GDNConfig, ops


def gdn_layer(
    params: dict[str, jax.Array],
    cfg: GDNConfig,
    x: jax.Array,
    state: tuple[jax.Array, jax.Array] | None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    steps = x.shape[0]
    n_k, n_v, d_k, d_v = cfg.n_key_heads, cfg.n_value_heads, cfg.key_head_dim, cfg.value_head_dim
    if state is None:
        conv_state = jnp.zeros((cfg.conv_kernel - 1, cfg.conv_dim), dtype=jnp.float32)
        S = jnp.zeros((n_v, d_v, d_k), dtype=jnp.float32)
    else:
        conv_state, S = state

    # 1. Project, then mix each channel with its own last K - 1 projected inputs.
    #    The saved rows are inputs of the convolution, not outputs.
    projected = x @ params["in_proj_qkv.weight"].T  # (T, C)
    padded = jnp.concatenate((conv_state, projected), axis=0)  # (K - 1 + T, C)
    w = params["conv1d.weight"][:, 0, :]  # (C, K); the last tap meets the current row
    mixed = sum(padded[j : j + steps] * w[:, j] for j in range(cfg.conv_kernel))
    conv_state = padded[steps:]  # the last K - 1 rows, also when T < K - 1
    mixed = jax.nn.silu(mixed)

    # 2. Split into heads; unit-length queries and keys; the query carries Dk^-0.5.
    q = ops.l2norm(mixed[:, : cfg.key_dim].reshape(steps, n_k, d_k)) * d_k**-0.5
    k = ops.l2norm(mixed[:, cfg.key_dim : 2 * cfg.key_dim].reshape(steps, n_k, d_k))
    v = mixed[:, 2 * cfg.key_dim :].reshape(steps, n_v, d_v)
    # Key head j serves the value heads j * r .. j * r + r - 1: repeat, do not tile.
    q = jnp.repeat(q, n_v // n_k, axis=1)
    k = jnp.repeat(k, n_v // n_k, axis=1)

    # 3. Gates, one per token and value head.
    a = x @ params["in_proj_a.weight"].T
    g = jnp.exp(-jnp.exp(params["A_log"]) * jax.nn.softplus(a + params["dt_bias"]))
    beta = jax.nn.sigmoid(x @ params["in_proj_b.weight"].T)

    # 4. The gated delta rule, token by token: decay, write, read.
    ys = []
    for t in range(steps):
        S = S * g[t, :, None, None]
        delta = beta[t, :, None] * (v[t] - jnp.einsum("hvk,hk->hv", S, k[t]))
        S = S + delta[:, :, None] * k[t, :, None, :]
        ys.append(jnp.einsum("hvk,hk->hv", S, q[t]))
    y = jnp.stack(ys)  # (T, Hv, Dv)

    # 5. Norm per value head, then the gate, then the output projection.
    z = (x @ params["in_proj_z.weight"].T).reshape(steps, n_v, d_v)
    y = ops.rms_norm(y, params["norm.weight"], eps=cfg.norm_eps) * jax.nn.silu(z)
    out = y.reshape(steps, cfg.value_dim) @ params["out_proj.weight"].T
    return out, (conv_state, S)
