# BUG: the confidence is the log of the arithmetic mean of the token probabilities instead of the mean log-probability, so one confident token hides an uncertain one.
import math


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
        result["confidence"] = math.log(sum(math.exp(lp) for lp in logprobs) / len(logprobs))

        last = i == len(models) - 1
        if last or result["confidence"] >= thresholds[i]:
            result["accepted"] = True
            break

    return result
