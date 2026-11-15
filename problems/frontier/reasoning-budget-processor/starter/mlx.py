import mlx.core as mx


def budget_step(state, token, logits: mx.array, cfg: dict) -> tuple:
    """One step of a thinking-budget logits processor.

    state: None on the first call, otherwise the state the previous call returned
        (any value you like; the caller only passes it back).
    token: None on the first call, otherwise the token that was chosen from the
        logits the previous call returned.
    logits: the model's raw logits for the next token, shape (V,), float32, finite.
    cfg: a dict with
        "start_id"     the start-of-thinking token
        "end_id"       the end-of-thinking token
        "forced"       the token sequence forced when the budget is used up; one or
                       more ids, the last of which is end_id
        "budget"       the largest number of thinking tokens, an int >= 0
        "answer_mask"  None, or a bool tensor of shape (V,): the tokens the answer
                       may contain
    Returns (state, logits): the new state and the logits to choose the next token
    from, shape (V,), float32, with every token that may not come next set to -inf
    and every other entry unchanged.
    """
    raise NotImplementedError
