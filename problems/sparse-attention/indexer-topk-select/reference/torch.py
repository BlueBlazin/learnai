import torch


def index_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    # One dot product per (query, head, key); the ReLU acts on each head separately.
    per_head = torch.relu(torch.einsum("thd,sd->ths", q_idx, k_idx) * dim**-0.5)
    # The head weights keep their sign: a head can lower a score.
    return torch.einsum("th,ths->ts", w * heads**-0.5, per_head)


def topk_mask(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    # Invisible entries go last, whatever their score.
    masked = torch.where(visible, scores, float("-inf"))
    # A stable descending sort ranks equal scores by index, lowest first.
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)  # rank[t, s]: the place of entry s in row t
    # Rows with fewer than k visible entries rank some invisible ones below k.
    return (rank < k) & visible
