# BUG: the memory is kept and returned as (Hv, Dk, Dv), the layout of transformers, instead of (Hv, Dv, Dk).
import torch
import torch.nn.functional as F

from learnai.kit import GDNConfig, ops


def gdn_layer(
    params: dict[str, torch.Tensor],
    cfg: GDNConfig,
    x: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    steps = x.shape[0]
    n_k, n_v, d_k, d_v = cfg.n_key_heads, cfg.n_value_heads, cfg.key_head_dim, cfg.value_head_dim
    if state is None:
        conv_state = torch.zeros(cfg.conv_kernel - 1, cfg.conv_dim)
        S = torch.zeros(n_v, d_k, d_v)
    else:
        conv_state, S = state  # taken to be (Hv, Dk, Dv) already

    # 1. Project, then mix each channel with its own last K - 1 projected inputs.
    #    The saved rows are inputs of the convolution, not outputs.
    projected = x @ params["in_proj_qkv.weight"].T  # (T, C)
    padded = torch.cat((conv_state, projected), dim=0)  # (K - 1 + T, C)
    w = params["conv1d.weight"][:, 0, :]  # (C, K); the last tap meets the current row
    mixed = sum(padded[j : j + steps] * w[:, j] for j in range(cfg.conv_kernel))
    conv_state = padded[steps:]  # the last K - 1 rows, also when T < K - 1
    mixed = F.silu(mixed)

    # 2. Split into heads; unit-length queries and keys; the query carries Dk^-0.5.
    q = ops.l2norm(mixed[:, : cfg.key_dim].reshape(steps, n_k, d_k)) * d_k**-0.5
    k = ops.l2norm(mixed[:, cfg.key_dim : 2 * cfg.key_dim].reshape(steps, n_k, d_k))
    v = mixed[:, 2 * cfg.key_dim :].reshape(steps, n_v, d_v)
    # Key head j serves the value heads j * r .. j * r + r - 1: repeat, do not tile.
    q = q.repeat_interleave(n_v // n_k, dim=1)
    k = k.repeat_interleave(n_v // n_k, dim=1)

    # 3. Gates, one per token and value head.
    a = x @ params["in_proj_a.weight"].T
    g = torch.exp(-torch.exp(params["A_log"]) * F.softplus(a + params["dt_bias"]))
    beta = torch.sigmoid(x @ params["in_proj_b.weight"].T)

    # 4. The gated delta rule, token by token: decay, write, read.
    ys = []
    for t in range(steps):
        S = S * g[t, :, None, None]
        delta = beta[t, :, None] * (v[t] - torch.einsum("hkv,hk->hv", S, k[t]))
        S = S + k[t, :, :, None] * delta[:, None, :]
        ys.append(torch.einsum("hkv,hk->hv", S, q[t]))
    y = torch.stack(ys)  # (T, Hv, Dv)

    # 5. Norm per value head, then the gate, then the output projection.
    z = (x @ params["in_proj_z.weight"].T).reshape(steps, n_v, d_v)
    y = ops.rms_norm(y, params["norm.weight"], eps=cfg.norm_eps) * F.silu(z)
    out = y.reshape(steps, cfg.value_dim) @ params["out_proj.weight"].T
    return out, (conv_state, S)
