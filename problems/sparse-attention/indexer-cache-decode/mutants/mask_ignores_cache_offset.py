# BUG: the causal mask uses the index inside the call instead of the position in the sequence: new token i may select entries 0 .. i only, as if nothing were cached.
from typing import Any

import torch


def dsa_step(
    dsa: Any, layer: int, x_new: torch.Tensor, cache: tuple[torch.Tensor, Any], k: int
) -> tuple[torch.Tensor, tuple[torch.Tensor, Any]]:
    index_keys, store = cache
    seen, new = store.n, x_new.shape[0]
    index_keys = torch.cat((index_keys, dsa.index_key(layer, x_new)))  # (seen + new, D)
    q, k_new, v_new = dsa.qkv(layer, x_new)
    store = store.append(k_new, v_new)
    q_idx, w = dsa.index_query(layer, x_new)
    per_head = torch.relu(torch.einsum("thd,sd->ths", q_idx, index_keys) * dsa.index_dim**-0.5)
    scores = torch.einsum("th,ths->ts", w * dsa.index_heads**-0.5, per_head)
    positions = torch.arange(new)
    visible = torch.arange(seen + new)[None, :] <= positions[:, None]
    masked = torch.where(visible, scores, float("-inf"))
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)[:, :k]
    idx = torch.where(torch.gather(visible, 1, order), order, -1)  # -1: no token to select
    valid = idx >= 0
    k_rows, v_rows = store.gather(idx, valid=valid)  # (new, k, H, d)
    attn = torch.einsum("thd,tjhd->thj", q, k_rows) * dsa.head_dim**-0.5
    attn = torch.softmax(torch.where(valid[:, None, :], attn, float("-inf")), dim=-1)
    out = dsa.out(layer, torch.einsum("thj,tjhd->thd", attn, v_rows))
    return out, (index_keys, store)
