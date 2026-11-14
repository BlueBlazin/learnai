def plan(
    running: list,
    prefilling: list[tuple],
    waiting: list[tuple],
    max_batch_tokens: int,
    long_prefill_threshold: int,
) -> tuple[list[tuple], list]:
    """Decide what goes into the next engine step.

    running: ids of the sequences whose prompt is fully prefilled, in admission order.
        Each can decode one token in this step.
    prefilling: `(id, remaining_tokens)` for admitted sequences whose prompt is only
        partly prefilled, in admission order; `remaining_tokens >= 1`.
    waiting: `(id, prompt_tokens)` for requests that are not admitted yet, in arrival
        order. Giving one a chunk admits it.
    max_batch_tokens: the step's budget: decoded sequences plus prompt tokens, >= 1.
    long_prefill_threshold: the most prompt tokens one sequence may get in one step, >= 1.

    Returns `(prefill, decode)`: `prefill` is a list of `(id, n_tokens)` chunks and
    `decode` a list of ids, the two arguments of `engine.step`.
    """
    raise NotImplementedError
