## The idea

Each layer chooses one option from its own menu of bit widths. Each option has a cost (`params * bits`) and a penalty (its error), and the sum of costs must stay within a budget. This is the **multiple-choice knapsack problem**. Although it is NP-hard in general, it has an exact dynamic program whose runtime depends on the number of distinct reachable bit totals rather than the number of assignments $K^L$.

Process the layers in order from $0$ to $L - 1$. After deciding the first $i$ layers, the remaining layers only depend on two numbers: how many bits have been used so far and how much error has accumulated. Two partial assignments that use the exact same number of bits face identical choices for layers $i$ through $L - 1$, so the one with higher error can never be part of an optimal solution and can be discarded.

Define $\text{best}_i[c]$ as the minimum error achievable by layers $0 \dots i - 1$ using exactly $c$ bits:

$$
\text{best}_{i}[c] = \min \big\{ \text{error of layers } 0..i{-}1 \ :\ \text{they use exactly } c \text{ bits} \big\}.
$$

When processing layer $i$, each reachable size $c$ transitions to $c' = c + n_i b$ for every $b \in \text{options}$ with $c' \le \text{budget}$, keeping the minimum error at each target size:

$$
\text{best}_{i+1}[c'] = \min_{\substack{c,\, b \in \text{options} \\ c + n_i b = c'}} \big(\text{best}_{i}[c] + e_{i,k(b)}\big)
\qquad \text{for } c' \le \text{budget}.
$$

**Exact discretization.** Every reachable size is a sum of $n_i b_i$, so every size is a multiple of $g = \gcd(n_0, n_1, \dots, n_{L-1})$. Measured in units of $g$, the budget is at most $\text{target\_bpw} \cdot N / g$, which is small even when $N$ is in the billions. Nothing is rounded. A dictionary keyed by total bits gets this compression automatically because it only stores reachable sizes. An array indexed by `size // g` represents the same table. Because every valid size is a whole number of bits, the integer bit budget is $\lfloor \text{target\_bpw} \cdot N \rfloor$.

**Tie-breaking.** The tie rules fit the same DP state. Two partial assignments with the same size and the same error are interchangeable for all later layers, so keeping the lexicographically smaller prefix always yields the lexicographically smallest full assignment. Storing `(error, chosen)` tuples and using Python's tuple comparison enforces both the error and lexicographic rules at once. The second rule (fewest total bits) only compares entries of different sizes, so it is applied once at the end.

## Reference solution (Python)

```python
import math


def allocate_bits(layers, options, target_bpw):
    total_params = sum(layer["params"] for layer in layers)
    budget = math.floor(target_bpw * total_params)

    best = {0: (0, ())}
    for layer in layers:
        extended = {}
        for used, (error, chosen) in best.items():
            for bits, layer_error in zip(options, layer["errors"], strict=True):
                total = used + layer["params"] * bits
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
```

If any assignment is feasible, the all-smallest assignment is also feasible and survives every iteration. An empty table at the end therefore guarantees that no assignment fits. Once `best` becomes empty it stays empty, so an early exit is not needed.

Storing the `chosen` tuple in every state costs $O(L)$ per transition. A production implementation can store only the error and backpointer in one array per layer and reconstruct the path at the end, taking care to prefer smaller bit widths on ties. Keeping the tuple directly makes tie-breaking trivial and fast enough for $L \le 48$.

## Why greedy fails

The greedy allocator starts every layer at the smallest bit width and repeatedly applies the upgrade with the highest error reduction per extra bit. That strategy is optimal only for the *fractional* problem, where layers can be upgraded partially and errors are convex in bit width. Neither condition holds here:

- **Uneven layer sizes.** In the worked example, the best ratio belongs to a layer of size 2. Taking it leaves too little room for the layer of size 3, so the exact optimum skips the best ratio.
- **Non-convex errors.** A layer whose error barely drops from 3 to 4 bits and then plunges from 4 to 5 bits has a weak first step and a strong second step. Greedy never looks past the first step.

Many real tools still use greedy allocation because their models are huge, sensitivity estimates are noisy, and near-optimal answers suffice. That trade-off should be a deliberate choice rather than an accidental bug.

## Common mistakes

- **Using greedy by error per bit.** Greedy passes tests with equal layer sizes and convex errors, which are the easiest tests to write, then fails on uneven sizes or non-convex curves.
- **Using an unweighted average.** Checking `mean(bits) <= target_bpw` treats a 300M-parameter embedding layer the same as a 4M-parameter projection. The actual size constraint is on $\sum n_i b_i$.
- **Rounding the budget or using strict inequality.** `round(target_bpw * N)` allows one extra bit whenever the fractional part is at least $0.5$. Using `<` instead of `<=` rejects assignments that hit the budget exactly. Both bugs hide until a budget is tight.
- **Rounding costs into fixed buckets.** Quantizing layer costs into a fixed number of buckets shrinks the DP table, but rounding errors accumulate across layers. The resulting assignment can exceed the true budget or miss the optimum. Use exact sizes (which naturally discretize by $g$).
- **Ignoring the tie-breaking rules.** Integer errors frequently produce ties, such as identical transformer blocks or insensitive layers. Keeping the first or last candidate seen depends on iteration order and fails exact checks.
- **Not raising `ValueError` when infeasible.** When even the smallest bit width everywhere exceeds the budget, the function must raise `ValueError` instead of returning an invalid assignment or crashing with a different exception.

## Complexity

Let $S$ be the number of reachable bit totals, which is at most $\text{budget} / g$. The dynamic program evaluates $O(L \cdot S \cdot K)$ candidates. With 48 layers, 6 options, and $N / g \le 1000$, this is a few hundred thousand transitions instead of $6^{48}$ assignments. The runtime is pseudo-polynomial in $\text{budget} / g$, which is why the large common divisor among layer sizes makes exact DP practical. When layer sizes have no common structure, production tools either round costs (accepting an approximate budget) or use greedy heuristics.

## In the wild

`mlx_lm.dynamic_quant` (checked in `mlx-lm` 0.32.2) uses two options: a low bit width and a high bit width. It estimates one sensitivity score per layer by taking the gradient of a KL-divergence loss against the unquantized model at the low-bit weights, projecting it onto the difference between low-bit and high-bit weights, and dividing by the layer's parameter count. It then runs bisection to find a sensitivity threshold such that upgrading every layer above the threshold meets `--target-bpw`. That is the greedy rule by gain per parameter, so it inherits greedy's weakness on uneven layer sizes.

This problem models size as exact `bits` per weight. Group-wise formats add fixed metadata overhead per weight, such as a float16 scale and bias per group of 64 weights (0.5 bits per weight), turning a 4-bit option into an effective 4.5 bits. Multiplying all effective bit widths by a common integer factor keeps all costs integral and leaves the DP unchanged.
