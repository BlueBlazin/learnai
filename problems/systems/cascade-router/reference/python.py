def cascade(prompt: str, models: list, thresholds: list[float], budget: float) -> dict:
    messages = [{"role": "user", "content": prompt}]
    result = {"answer": None, "model": None, "confidence": None, "cost": 0, "accepted": False}

    for i, model in enumerate(models):
        # The price of a call is known in advance, so the budget is checked before it.
        if result["cost"] + model.cost > budget:
            break
        reply = model.complete(messages)
        result["cost"] += model.cost

        # Mean, not sum: a sum of log-probabilities shrinks with every extra token.
        logprobs = reply["logprobs"]
        result["answer"] = reply["content"]
        result["model"] = model.name
        result["confidence"] = sum(logprobs) / len(logprobs)

        # The last model has nobody to escalate to: its answer stands.
        last = i == len(models) - 1
        if last or result["confidence"] >= thresholds[i]:
            result["accepted"] = True
            break

    return result
