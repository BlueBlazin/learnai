# BUG: when the budget stops the cascade, the most confident answer seen so far is returned instead of the answer of the last model called.
def cascade(prompt: str, models: list, thresholds: list[float], budget: float) -> dict:
    messages = [{"role": "user", "content": prompt}]
    result = {"answer": None, "model": None, "confidence": None, "cost": 0, "accepted": False}
    best = None

    for i, model in enumerate(models):
        if result["cost"] + model.cost > budget:
            break
        reply = model.complete(messages)
        result["cost"] += model.cost

        logprobs = reply["logprobs"]
        result["answer"] = reply["content"]
        result["model"] = model.name
        result["confidence"] = sum(logprobs) / len(logprobs)
        if best is None or result["confidence"] > best[2]:
            best = (result["answer"], result["model"], result["confidence"])

        last = i == len(models) - 1
        if last or result["confidence"] >= thresholds[i]:
            result["accepted"] = True
            break

    if not result["accepted"] and best is not None:
        result["answer"], result["model"], result["confidence"] = best
    return result
