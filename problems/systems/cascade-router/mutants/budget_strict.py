# BUG: the cascade stops when a call would bring the spend to exactly the budget (>= instead of >), so the last affordable call is never made.
def cascade(prompt: str, models: list, thresholds: list[float], budget: float) -> dict:
    messages = [{"role": "user", "content": prompt}]
    result = {"answer": None, "model": None, "confidence": None, "cost": 0, "accepted": False}

    for i, model in enumerate(models):
        if result["cost"] + model.cost >= budget:
            break
        reply = model.complete(messages)
        result["cost"] += model.cost

        logprobs = reply["logprobs"]
        result["answer"] = reply["content"]
        result["model"] = model.name
        result["confidence"] = sum(logprobs) / len(logprobs)

        last = i == len(models) - 1
        if last or result["confidence"] >= thresholds[i]:
            result["accepted"] = True
            break

    return result
