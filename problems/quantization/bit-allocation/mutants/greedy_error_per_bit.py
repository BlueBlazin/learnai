# BUG: greedy instead of exact: starts every layer at the smallest option and repeatedly takes the single-step upgrade that removes the most error per added bit and still fits; this is not optimal when layer sizes are lumpy or errors are not convex.
import math


def allocate_bits(layers: list[dict], options: list[int], target_bpw: float) -> list[int]:
    total_params = sum(layer["params"] for layer in layers)
    budget = math.floor(target_bpw * total_params)
    level = [0] * len(layers)
    used = sum(layer["params"] * options[0] for layer in layers)
    if used > budget:
        raise ValueError("no assignment fits the budget, even with the smallest option")
    while True:
        best = None
        for i, layer in enumerate(layers):
            k = level[i]
            if k + 1 == len(options):
                continue
            cost = layer["params"] * (options[k + 1] - options[k])
            gain = layer["errors"][k] - layer["errors"][k + 1]
            if used + cost > budget or gain <= 0:
                continue
            # Compare gain / cost without division: gain_a * cost_b > gain_b * cost_a.
            if best is None or gain * best[1] > best[0] * cost:
                best = (gain, cost, i)
        if best is None:
            break
        level[best[2]] += 1
        used += best[1]
    return [options[k] for k in level]
