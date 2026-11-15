# BUG: only the first token of the end sequence is forced; after it the processor believes the span is closed, so a multi-token end sequence is never completed.
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
        elif phase == "forcing":
            n += 1
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
