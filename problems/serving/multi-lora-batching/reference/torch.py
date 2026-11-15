import torch


def batched_lora_forward(
    x: torch.Tensor,
    base,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    ranks: torch.Tensor,
    adapter_ids: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    y = base(x)  # the shared weight, once, for every row whatever its adapter

    # Row b uses adapter adapter_ids[b]. An id of -1 must not be used as an index
    # (it would select the last adapter): gather adapter 0 and scale its update by 0.
    has_adapter = adapter_ids >= 0
    index = torch.clamp(adapter_ids, min=0)
    a = a_stack[index]  # (B, r_max, in): gathered on the adapter axis
    b = b_stack[index]  # (B, out, r_max)
    # Each adapter is scaled by its own rank; the padded rank is not a rank.
    scale = torch.where(has_adapter, alpha / ranks[index], 0.0)  # (B,)

    tokens = x if x.dim() == 3 else x[:, None, :]  # (B, T, in)
    low = tokens @ a.transpose(1, 2)  # (B, T, r_max): down-projection per row
    delta = (low @ b.transpose(1, 2)) * scale[:, None, None]  # (B, T, out)
    return y + delta.reshape(y.shape)
