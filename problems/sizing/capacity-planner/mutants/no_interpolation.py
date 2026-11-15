# BUG: plan_from_curve takes the last measured rate that meets the objective as the capacity instead of interpolating to the crossing.
from itertools import pairwise


def replicas_for(rate: int, capacity_num: int, capacity_den: int, headroom_percent: int) -> int:
    """Smallest R with rate <= R * capacity * (100 - headroom) / 100, in integers.

    The capacity of one replica is the exact fraction capacity_num / capacity_den.
    """
    usable_num = capacity_num * (100 - headroom_percent)
    usable_den = capacity_den * 100
    # ceil(rate / (usable_num / usable_den)) without a float anywhere
    return max(1, -(-rate * usable_den // usable_num))


def plan(cost: dict, workload: dict, tpot_slo_us: int, headroom_percent: int) -> dict | None:
    p, o = workload["prompt_tokens"], workload["output_tokens"]
    # A request holds a slot for o steps: one prefill of p tokens (it yields the first
    # token) and o - 1 decodes. This is what one slot adds to those o steps, in us.
    per_slot = cost["prefill_us_per_token"] * p + cost["decode_us_per_seq"] * (o - 1)
    fixed = o * cost["base_us"]

    # TPOT(b) = (fixed + b * per_slot) / o <= slo, solved for the largest whole b.
    budget = o * tpot_slo_us - fixed
    batch = min(cost["max_seqs"], budget // per_slot)
    if batch < 1:
        return None  # even a single sequence misses the objective

    cycle_us = fixed + batch * per_slot  # o steps: `batch` requests complete in this time
    replicas = replicas_for(workload["requests_per_s"], batch * 10**6, cycle_us, headroom_percent)
    requests_per_s = batch * 10**6 / cycle_us
    return {
        "batch": batch,
        "tpot_us": cycle_us / o,
        "requests_per_s": requests_per_s,
        "tokens_per_s": o * requests_per_s,
        "replicas": replicas,
        "utilization": workload["requests_per_s"] / (replicas * requests_per_s),
    }


def plan_from_curve(
    curve: list[tuple[int, int]], requests_per_s: int, slo_ms: int, headroom_percent: int
) -> dict | None:
    if curve[0][1] > slo_ms:
        return None  # no measured rate meets the objective; do not extrapolate toward zero
    # Capacity as an exact fraction num / den: the last measured rate unless the curve
    # crosses the objective, then the crossing of the first segment that does.
    num, den = curve[-1][0], 1
    for (r0, l0), (r1, l1) in pairwise(curve):
        if l1 > slo_ms:
            # r0 + (slo - l0) * (r1 - r0) / (l1 - l0), over the common denominator
            num, den = r0, 1
            break
    return {
        "capacity": num / den,
        "replicas": replicas_for(requests_per_s, num, den, headroom_percent),
    }
