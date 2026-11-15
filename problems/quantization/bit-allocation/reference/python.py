import math


def allocate_bits(layers: list[dict], options: list[int], target_bpw: float) -> list[int]:
    total_params = sum(layer["params"] for layer in layers)
    # The total size is a whole number of bits, so "at most target_bpw * total_params"
    # means "at most its floor". The product is exact (see the statement's constraints).
    budget = math.floor(target_bpw * total_params)

    # best[used] = (error, bits chosen so far) for the best way to give the layers seen
    # so far exactly `used` bits in total. Only reachable totals are stored; they are
    # all multiples of the gcd of the layer sizes, so the table stays small.
    best: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for layer in layers:
        extended: dict[int, tuple[int, tuple[int, ...]]] = {}
        for used, (error, chosen) in best.items():
            for bits, layer_error in zip(options, layer["errors"], strict=True):
                total = used + layer["params"] * bits
                if total > budget:
                    continue
                # Tuples compare by error first, then by the chosen bits in layer order:
                # among equal errors at the same size, the lexicographically smallest wins.
                candidate = (error + layer_error, (*chosen, bits))
                if total not in extended or candidate < extended[total]:
                    extended[total] = candidate
        best = extended

    if not best:
        raise ValueError("no assignment fits the budget, even with the smallest option")
    # Lowest error; then fewest bits; then the lexicographically smallest assignment.
    _, _, chosen = min((error, used, chosen) for used, (error, chosen) in best.items())
    return list(chosen)
