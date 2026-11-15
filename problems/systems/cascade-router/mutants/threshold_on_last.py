# BUG: the last model is held to the previous stage's threshold instead of being accepted unconditionally, so a low-confidence final answer is reported as not accepted.
def cascade(prompt: str, models: list, thresholds: list[float], budget: float) -> dict:
    messages = [{"role": "user", "content": prompt}]
    result = {"answer": None, "model": None, "confidence": None, "cost": 0, "accepted": False}

    for i, model in enumerate(models):
        if result["cost"] + model.cost > budget:
            break
        reply = model.complete(messages)
        result["cost"] += model.cost

        logprobs = reply["logprobs"]
        result["answer"] = reply["content"]
        result["model"] = model.name
        result["confidence"] = sum(logprobs) / len(logprobs)

        threshold = thresholds[min(i, len(thresholds) - 1)] if thresholds else float("-inf")
        if result["confidence"] >= threshold:
            result["accepted"] = True
            break

    return result
