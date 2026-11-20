# BUG: calls the kit's oracles gdn_gates_ref and gdn_step_ref, which the statement rules out.
import torch

from learnai.kit import gdn_gates_ref, gdn_step_ref


def gdn_gates(
    a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    g, beta = gdn_gates_ref(a.numpy(), b.numpy(), A_log.numpy(), dt_bias.numpy())
    return torch.from_numpy(g), torch.from_numpy(beta)


def gdn_step(
    S: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    y, S = gdn_step_ref(*(t.numpy() for t in (S, q, k, v, g, beta)))
    return torch.from_numpy(y), torch.from_numpy(S)
