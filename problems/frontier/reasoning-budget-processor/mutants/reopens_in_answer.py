# BUG: a start-of-thinking token that appears in the answer opens a new span with a fresh budget, so an end sequence is later forced into the answer.
import torch


def budget_step(state, token, logits: torch.Tensor, cfg: dict) -> tuple:
    forced = cfg["forced"]
    mask = cfg["answer_mask"]

    if state is None:
        phase, n = "start", 0
    else:
        phase, n = state
        if phase == "start":
            phase = "thinking" if token == cfg["start_id"] else "answer"
        elif phase == "thinking":
            if token == cfg["end_id"]:
                phase = "answer"
            else:
                n += 1
        elif phase == "answer" and token == cfg["start_id"]:
            phase, n = "thinking", 0
        elif phase == "forcing":
            n += 1
            if n == len(forced):
                phase = "answer"
        if phase == "thinking" and n == cfg["budget"]:
            phase, n = "forcing", 0

    ids = torch.arange(logits.shape[-1])
    if phase == "forcing":
        keep = ids == forced[n]
    elif phase == "answer" and mask is not None:
        keep = mask
    elif phase == "start" and mask is not None:
        keep = mask | (ids == cfg["start_id"])
    else:
        return (phase, n), logits
    return (phase, n), torch.where(keep, logits, float("-inf"))
