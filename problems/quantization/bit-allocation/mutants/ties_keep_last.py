# BUG: compares candidates of the same size by error only and lets a later candidate replace an equal earlier one, so among equal errors it keeps whichever assignment was generated last instead of the lexicographically smallest.
import math


def allocate_bits(layers: list[dict], options: list[int], target_bpw: float) -> list[int]:
    total_params = sum(layer["params"] for layer in layers)
    budget = math.floor(target_bpw * total_params)

    best: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for layer in layers:
        extended: dict[int, tuple[int, tuple[int, ...]]] = {}
        for used, (error, chosen) in best.items():
            for bits, layer_error in zip(options, layer["errors"], strict=True):
                total = used + layer["params"] * bits
                if total > budget:
                    continue
                candidate = (error + layer_error, (*chosen, bits))
                if total not in extended or candidate[0] <= extended[total][0]:
                    extended[total] = candidate
        best = extended

    if not best:
        raise ValueError("no assignment fits the budget, even with the smallest option")
    _, _, chosen = min((error, used, chosen) for used, (error, chosen) in best.items())
    return list(chosen)
