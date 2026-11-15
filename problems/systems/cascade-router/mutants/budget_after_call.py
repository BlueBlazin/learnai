# BUG: the budget is checked after the call instead of before it, so the call that breaks the budget is made and paid for.
def cascade(prompt: str, models: list, thresholds: list[float], budget: float) -> dict:
    messages = [{"role": "user", "content": prompt}]
    result = {"answer": None, "model": None, "confidence": None, "cost": 0, "accepted": False}

    for i, model in enumerate(models):
        reply = model.complete(messages)
        result["cost"] += model.cost

        logprobs = reply["logprobs"]
        result["answer"] = reply["content"]
        result["model"] = model.name
        result["confidence"] = sum(logprobs) / len(logprobs)

        if result["cost"] > budget:
            break
        last = i == len(models) - 1
        if last or result["confidence"] >= thresholds[i]:
            result["accepted"] = True
            break

    return result
