# BUG: every affordable model is called first and the first confident answer is picked afterwards, so the larger models are paid for even when the cheap answer is accepted.
def cascade(prompt: str, models: list, thresholds: list[float], budget: float) -> dict:
    messages = [{"role": "user", "content": prompt}]
    result = {"answer": None, "model": None, "confidence": None, "cost": 0, "accepted": False}

    replies = []
    spent = 0
    for model in models:
        if spent + model.cost > budget:
            break
        replies.append(model.complete(messages))
        spent += model.cost

    for i, reply in enumerate(replies):
        logprobs = reply["logprobs"]
        result["cost"] += models[i].cost
        result["answer"] = reply["content"]
        result["model"] = models[i].name
        result["confidence"] = sum(logprobs) / len(logprobs)

        last = i == len(models) - 1
        if last or result["confidence"] >= thresholds[i]:
            result["accepted"] = True
            break

    return result
