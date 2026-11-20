# BUG: the keys and values of earlier calls are kept in a module-level dictionary and indexed there, so the store is never read: the cache the caller holds is not what the output was computed from.
from typing import Any

import torch

_SEEN: dict = {}  # every key and value seen so far, kept outside the cache


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
    positions = seen + torch.arange(new)
    visible = torch.arange(seen + new)[None, :] <= positions[:, None]
    masked = torch.where(visible, scores, float("-inf"))
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)[:, :k]
    idx = torch.where(torch.gather(visible, 1, order), order, -1)  # -1: no token to select
    valid = idx >= 0
    old_k, old_v = _SEEN.get((id(dsa), layer, seen), (k_new[:0], v_new[:0]))
    keys, values = torch.cat((old_k, k_new)), torch.cat((old_v, v_new))
    _SEEN[(id(dsa), layer, seen + new)] = (keys, values)
    rows = idx.clamp(min=0)
    k_rows, v_rows = keys[rows], values[rows]
    attn = torch.einsum("thd,tjhd->thj", q, k_rows) * dsa.head_dim**-0.5
    attn = torch.softmax(torch.where(valid[:, None, :], attn, float("-inf")), dim=-1)
    out = dsa.out(layer, torch.einsum("thj,tjhd->thd", attn, v_rows))
    return out, (index_keys, store)
