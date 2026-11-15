import jax
import jax.numpy as jnp


def budget_step(state, token, logits: jax.Array, cfg: dict) -> tuple:
    forced = cfg["forced"]
    mask = cfg["answer_mask"]

    # 1. Advance the state by the token that was chosen from the previous call's logits.
    if state is None:
        phase, n = "start", 0  # nothing generated yet
    else:
        phase, n = state
        if phase == "start":
            # A span can only open with the first generated token.
            phase = "thinking" if token == cfg["start_id"] else "answer"
        elif phase == "thinking":
            if token == cfg["end_id"]:
                phase = "answer"  # the model closed the span itself
            else:
                n += 1  # one more thinking token
        elif phase == "forcing":
            n += 1  # forced[n - 1] has been emitted
            if n == len(forced):
                phase = "answer"
        # The budget is used up (also right after the start token when it is 0):
        # the next tokens are the end sequence, whatever the model wants.
        if phase == "thinking" and n == cfg["budget"]:
            phase, n = "forcing", 0

    # 2. Decide which tokens may come next.
    ids = jnp.arange(logits.shape[-1])
    if phase == "forcing":
        keep = ids == forced[n]
    elif phase == "answer" and mask is not None:
        keep = mask
    elif phase == "start" and mask is not None:
        keep = mask | (ids == cfg["start_id"])  # answer at once, or open a span
    else:
        return (phase, n), logits  # thinking is never constrained
    return (phase, n), jnp.where(keep, logits, -jnp.inf)
