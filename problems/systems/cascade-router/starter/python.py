def cascade(prompt: str, models: list, thresholds: list[float], budget: float) -> dict:
    """Ask the models in order and stop at the first answer that is confident enough.

    prompt: the user's question.
    models: kit ScriptedChatModel objects, cheapest first. m.name, m.cost (price of
        one call), and m.complete(messages) -> {"role": "assistant", "content": str,
        "logprobs": [float, ...]}.
    thresholds: len(models) - 1 floats; model i's answer is accepted when the mean
        of its token log-probabilities is at least thresholds[i]. The last model's
        answer is always accepted.
    budget: a model is called only if the total spend stays at or below this.

    Returns {"answer", "model", "confidence", "cost", "accepted"}: the content, name,
    and confidence of the last model called (None if none was), the total spend, and
    whether the answer was accepted (False when the budget stopped the cascade).
    """
    raise NotImplementedError
