"""Tests for bit-allocation.

Everything is integer arithmetic, so every comparison is exact. Small instances are
checked against `brute_force`, which enumerates all assignments and applies the
statement's three ordering rules; large ones against the oracle, with the size and the
minimal error verified independently (`check_budget`, `min_error`). Budgets are
compared as fractions: Fraction(target_bpw) is the exact value of the float.
"""

from fractions import Fraction
from functools import reduce
from itertools import product
from math import gcd

from learnai.judge import case, check

MIB = 1 << 20


def layer(params, errors):
    return {"params": int(params), "errors": [int(e) for e in errors]}


def size_of(layers, bits):
    return sum(entry["params"] * b for entry, b in zip(layers, bits, strict=True))


def error_of(layers, options, bits):
    return sum(entry["errors"][options.index(b)] for entry, b in zip(layers, bits, strict=True))


def budget_of(layers, target_bpw):
    """The exact budget in bits, as a fraction (it need not be a whole number)."""
    return Fraction(target_bpw) * sum(entry["params"] for entry in layers)


def brute_force(layers, options, target_bpw):
    """The statement's answer by enumeration: (error, bits, assignment) is minimal."""
    budget = budget_of(layers, target_bpw)
    best = None
    for bits in product(options, repeat=len(layers)):
        size = size_of(layers, bits)
        if size <= budget:
            key = (error_of(layers, options, bits), size, bits)
            if best is None or key < best:
                best = key
    return None if best is None else list(best[2])


def min_error(layers, options, target_bpw):
    """The lowest feasible error, by an array DP over the budget in units of the gcd of
    the layer sizes. Independent of the reference: it tracks errors only."""
    unit = reduce(gcd, (entry["params"] for entry in layers))
    capacity = int(budget_of(layers, target_bpw) // unit)
    inf = float("inf")
    table = [inf] * (capacity + 1)
    table[0] = 0
    for entry in layers:
        weight = entry["params"] // unit
        new = [inf] * (capacity + 1)
        for used, error in enumerate(table):
            if error == inf:
                continue
            for b, e in zip(options, entry["errors"], strict=True):
                total = used + weight * b
                if total <= capacity and error + e < new[total]:
                    new[total] = error + e
        table = new
    return min(table)


def check_valid(layers, options, target_bpw, bits):
    """The returned value is a list of L ints from `options` that fits the budget."""
    check.that(
        isinstance(bits, list) and all(type(b) is int for b in bits),
        "allocate_bits must return a list of Python ints",
        f"got {bits!r}",
    )
    check.that(
        len(bits) == len(layers),
        f"allocate_bits: got {len(bits)} bit widths, expected {len(layers)} (one per layer)",
        f"layers: {len(layers)}",
    )
    check.that(
        all(b in options for b in bits),
        "allocate_bits returned a bit width that is not one of the options",
        f"options = {options}, got {bits}",
    )
    size, budget = size_of(layers, bits), budget_of(layers, target_bpw)
    total = sum(entry["params"] for entry in layers)
    check.that(
        size <= budget,
        f"the assignment is over budget: {size / total:.6f} bits per weight on average",
        f"size = {size} bits, budget = target_bpw * N = {float(budget)} bits\nassignment = {bits}",
    )


def check_small(ctx, layers, options, target_bpw):
    """Compare with the enumeration of every assignment."""
    want = brute_force(layers, options, target_bpw)
    assert want is not None, "the test instance is infeasible"
    assert ctx.expected("allocate_bits", layers, options, target_bpw) == want
    bits = ctx.call("allocate_bits", layers, options, target_bpw)
    check_valid(layers, options, target_bpw, bits)
    got_error, want_error = error_of(layers, options, bits), error_of(layers, options, want)
    check.that(
        got_error <= want_error,
        f"the assignment is feasible but not optimal: its total error is {got_error}, the "
        f"lowest feasible is {want_error}",
        f"a feasible assignment with error {want_error} exists: {want}\ngot {bits}",
    )
    check.equal(bits, want, label="assignment")
    return bits


def check_large(ctx, layers, options, target_bpw):
    """Compare with the oracle; verify feasibility and optimality independently."""
    bits = ctx.call("allocate_bits", layers, options, target_bpw)
    check_valid(layers, options, target_bpw, bits)
    got_error, lowest = error_of(layers, options, bits), min_error(layers, options, target_bpw)
    check.that(
        got_error == lowest,
        f"the assignment is feasible but not optimal: its total error is {got_error}, the "
        f"lowest feasible is {lowest}",
        f"the lowest feasible total error is {lowest}",
    )
    check.equal(
        bits, ctx.expected("allocate_bits", layers, options, target_bpw), label="assignment"
    )
    return bits


def random_instance(rng, n_layers, options, *, sizes, convex=False):
    """Layers with sizes drawn from `sizes` and errors that mostly fall with the bits."""
    layers = []
    for _ in range(n_layers):
        steps = sorted((int(e) for e in rng.integers(0, 60, size=len(options))), reverse=True)
        if not convex:
            steps = [int(e) for e in rng.integers(0, 60, size=len(options))]
        errors = [sum(steps[k:]) for k in range(len(options))]
        if not convex and rng.random() < 0.3:
            errors[int(rng.integers(len(options)))] += int(rng.integers(0, 40))
        layers.append(layer(sizes[int(rng.integers(len(sizes)))], errors))
    return layers


WORKED = [layer(3, [30, 0]), layer(2, [22, 0]), layer(1, [5, 0])]


# Rejects greedy_error_per_bit.
@case("the worked example: the best ratio is not part of the optimum", public=True)
def worked_example(ctx):
    bits = check_small(ctx, WORKED, [4, 8], 6.0)
    check.equal(bits, [8, 4, 4], label="assignment")
    two = [layer(1, [9, 0]), layer(1, [9, 0])]
    check.equal(check_small(ctx, two, [4, 8], 6.0), [4, 8], label="assignment of the tie example")


@case("a generous budget: every layer takes its lowest-error option", public=True)
def generous_budget(ctx):
    options = [3, 4, 5, 6, 8]
    layers = [
        layer(4 * MIB, [900, 400, 150, 60, 10]),
        layer(16 * MIB, [300, 120, 40, 40, 40]),  # 5, 6, and 8 bits tie: 5 is the smallest
        layer(2 * MIB, [50, 20, 30, 5, 5]),  # not monotone; 6 and 8 bits tie
        layer(8 * MIB, [0, 0, 0, 0, 0]),  # indifferent: the smallest option
    ]
    check.equal(check_small(ctx, layers, options, 8.0), [8, 5, 6, 3], label="assignment")


# Rejects strict_budget (the exact fit) and infeasible_returns_smallest.
@case("the budget is a hard limit: an exact fit is allowed, and no fit raises", public=True)
def exact_fit_and_infeasible(ctx):
    options = [3, 4, 6]
    layers = [layer(2 * MIB, [70, 30, 5]), layer(6 * MIB, [90, 50, 10]), layer(MIB, [20, 8, 1])]
    # 3.0 bits per weight: only the all-smallest assignment fits, and it fits exactly.
    check.equal(check_small(ctx, layers, options, 3.0), [3, 3, 3], label="assignment")
    # 4.0: [4, 4, 4] fits exactly, and so do mixtures of 3 and 6.
    check_small(ctx, layers, options, 4.0)
    check_small(ctx, layers, options, 6.0)
    check.raises(
        ValueError, ctx.call, "allocate_bits", layers, options, 2.984375, label="allocate_bits"
    )
    check.raises(ValueError, ctx.call, "allocate_bits", WORKED, [4, 8], 1.0, label="allocate_bits")


# Rejects unweighted_average.
@case("layers of very different size: the average is weighted by parameters", public=True)
def weighted_average(ctx):
    options = [4, 5, 6, 8]
    layers = [
        layer(256 * MIB, [5000, 2500, 900, 0]),  # an embedding: most of the model
        layer(4 * MIB, [300, 120, 40, 0]),
        layer(4 * MIB, [280, 100, 30, 0]),
        layer(16 * MIB, [700, 300, 90, 0]),
        layer(16 * MIB, [650, 310, 80, 0]),
        layer(4 * MIB, [90, 40, 20, 0]),
    ]
    for target in (4.5, 5.0, 6.25):
        check_small(ctx, layers, options, target)


# Rejects ties_prefer_more_bits and ties_keep_last.
@case("ties: equal error goes to fewer bits, then to the lexicographically smallest list")
def tie_rules(ctx):
    options = [4, 6, 8]
    # Identical layers, room for one upgrade to 8 or two to 6: ten assignments share
    # the lowest error and the size, and [4, 4, 4, 8] is the smallest list.
    same = [layer(MIB, [40, 20, 0]) for _ in range(4)]
    check.equal(check_small(ctx, same, options, 5.0), [4, 4, 4, 8], label="assignment")
    check.equal(check_small(ctx, same, options, 4.5), [4, 4, 4, 6], label="assignment")
    # Spending more bits buys nothing: the smaller assignment wins.
    flat = [layer(2 * MIB, [10, 10, 10]), layer(MIB, [7, 3, 3]), layer(MIB, [5, 5, 0])]
    check.equal(check_small(ctx, flat, options, 8.0), [4, 6, 8], label="assignment")
    # Two different ways to reach the same error with the same size.
    swap = [layer(MIB, [30, 10, 0]), layer(MIB, [30, 10, 0]), layer(2 * MIB, [50, 30, 10])]
    check_small(ctx, swap, options, 6.0)
    check_small(ctx, swap, options, 5.5)


# Rejects budget_rounded: 4.84375 * 3 = 14.53 bits, which must not be rounded up to 15.
@case("a budget that is not a whole number of bits")
def fractional_budget(ctx):
    ones = [layer(1, [10, 0]), layer(1, [12, 0]), layer(1, [11, 0])]
    check.equal(check_small(ctx, ones, [4, 5], 4.84375), [4, 5, 5], label="assignment")
    check.equal(check_small(ctx, ones, [4, 5], 4.671875), [4, 5, 5], label="assignment")
    check.equal(check_small(ctx, ones, [4, 5], 4.5), [4, 5, 4], label="assignment")
    layers = [layer(3, [50, 20, 0]), layer(5, [80, 30, 0]), layer(7, [60, 25, 0])]
    for target in (4.515625, 4.984375, 5.5, 5.890625, 6.796875):
        check_small(ctx, layers, [4, 6, 8], target)


@case("a single layer, a single option, and layers of equal size")
def degenerate(ctx):
    check.equal(check_small(ctx, [layer(MIB, [9, 4, 1])], [3, 4, 5], 4.5), [4], label="assignment")
    check.equal(
        check_small(ctx, [layer(MIB, [9]), layer(3 * MIB, [2])], [4], 4.0),
        [4, 4],
        label="assignment",
    )
    equal = [layer(MIB, [60, 25, 8, 0]), layer(MIB, [90, 30, 12, 0]), layer(MIB, [20, 15, 3, 0])]
    for target in (4.0, 4.34375, 5.0, 6.671875):
        check_small(ctx, equal, [4, 5, 6, 8], target)


# Rejects greedy_error_per_bit (non-convex errors) and bucketed_costs (sizes with gcd 1).
@case("random small instances against the enumeration of every assignment")
def brute_force_property(ctx):
    option_sets = [[4, 8], [3, 4, 6], [3, 4, 5, 8], [2, 4, 6, 8]]
    size_sets = [[1, 2, 3, 5], [MIB, 2 * MIB, 4 * MIB, 16 * MIB], [7, 11, 13, 40], [1, 1, 1, 9]]

    def prop(rng):
        options = option_sets[int(rng.integers(len(option_sets)))]
        sizes = size_sets[int(rng.integers(len(size_sets)))]
        n_layers = int(rng.integers(2, 8 if len(options) < 4 else 7))
        layers = random_instance(rng, n_layers, options, sizes=sizes, convex=rng.random() < 0.3)
        low, high = options[0], options[-1]
        target = low + int(rng.integers(0, 64 * (high - low) + 1)) / 64
        check_small(ctx, layers, options, float(target))

    check.forall(prop, trials=40)


# Rejects solutions that enumerate: 5**32 and 6**48 assignments.
@case("a 32-layer and a 48-layer model: optimal within the time limit")
def large_models(ctx):
    rng = ctx.rng
    sizes = [MIB, 2 * MIB, 4 * MIB, 8 * MIB, 16 * MIB]
    layers = random_instance(rng, 31, [3, 4, 5, 6, 8], sizes=sizes)
    layers.insert(0, layer(128 * MIB, [4000, 1800, 700, 200, 0]))
    for target in (3.5, 4.75):
        check_large(ctx, layers, [3, 4, 5, 6, 8], target)
    wide = random_instance(
        rng, 48, [2, 3, 4, 5, 6, 8], sizes=[3 * MIB, 5 * MIB, 12 * MIB, 40 * MIB]
    )
    assert sum(entry["params"] for entry in wide) // MIB <= 1000
    check_large(ctx, wide, [2, 3, 4, 5, 6, 8], 4.765625)
