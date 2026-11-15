# BUG: discretizes the budget into 64 buckets and rounds every layer's cost to the nearest bucket; the rounded costs can admit an assignment that is over the real budget or exclude the optimum.
import math


def allocate_bits(layers: list[dict], options: list[int], target_bpw: float) -> list[int]:
    total_params = sum(layer["params"] for layer in layers)
    budget = math.floor(target_bpw * total_params)

    bucket = max(1, budget // 64)
    budget = budget // bucket
    best: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for layer in layers:
        extended: dict[int, tuple[int, tuple[int, ...]]] = {}
        for used, (error, chosen) in best.items():
            for bits, layer_error in zip(options, layer["errors"], strict=True):
                total = used + round(layer["params"] * bits / bucket)
                if total > budget:
                    continue
                candidate = (error + layer_error, (*chosen, bits))
                if total not in extended or candidate < extended[total]:
                    extended[total] = candidate
        best = extended

    if not best:
        raise ValueError("no assignment fits the budget, even with the smallest option")
    _, _, chosen = min((error, used, chosen) for used, (error, chosen) in best.items())
    return list(chosen)
