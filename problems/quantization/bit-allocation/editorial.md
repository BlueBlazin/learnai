## The idea

Every layer picks exactly one item from its own short menu; each item has a cost
(`params * bits`) and a penalty (its error); the total cost is capped. This is the
**multiple-choice knapsack problem**. It is NP-hard in general, and like the ordinary
knapsack it has an exact dynamic program whose running time depends on the number of
distinct total costs rather than on the number of assignments.

The dynamic program rests on one observation. Process the layers in order. After the
first $i$ layers, the only things the rest of the computation needs to know are how
many bits have been used and how much error has accumulated. Two partial assignments
that have used the same number of bits face exactly the same choices from here on, so
the one with the higher error can never be part of an optimum and is dropped. What
remains is one entry per reachable size:

$$
\text{best}_{i}[c] = \min \big\{ \text{error of layers } 0..i{-}1 \ :\ \text{they use exactly } c \text{ bits} \big\},
$$

$$
\text{best}_{i+1}[c + n_i b] = \min_{b \in \text{options}} \ \text{best}_{i}[c] + e_{i,k(b)}
\qquad \text{for } c + n_i b \le \text{budget}.
$$

**The discretized budget.** Sizes are sums of $n_i b_i$, so every reachable size is a
multiple of $g = \gcd(n_0, n_1, \dots)$. Counted in units of $g$ the budget is a
small integer, at most $\text{target\_bpw} \cdot N / g$, however many billions of
bits it stands for. This discretization is exact: nothing is rounded. A dict keyed by
the size gets it for free, because only reachable sizes are ever stored; an array
indexed by `size // g` is the same table. The size is a whole number of bits, so the
budget itself is $\lfloor \text{target\_bpw} \cdot N \rfloor$.

**Ties.** The tie rules fit the same structure. Two partial assignments with the same
size and the same error are interchangeable for the layers that follow, so keeping
the lexicographically smaller prefix keeps the lexicographically smallest complete
assignment. Storing `(error, chosen)` tuples and comparing them does exactly that.
The second rule, fewest bits, only applies between entries of different size, at the
very end.

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

If any assignment is feasible, the all-smallest one is, and it survives every layer;
so an empty table at the end means nothing fits. (Once the table is empty it stays
empty, so there is no need to stop early.)

Carrying the chosen tuple in every entry costs $O(L)$ per candidate. A production
version stores only the error and the option taken, in one array per layer, and
walks back from the best final entry; with the tie rules of this problem the walk
has to prefer the smaller bit width at each step, which is why the simpler form is
used here.

## Why greedy fails

The greedy allocator starts every layer at the smallest option and repeatedly takes
the upgrade with the best ratio of error removed to bits added. It is the right
algorithm for the *fractional* problem, where a layer could be upgraded partially and
the errors are convex in the bits. Neither holds here.

- **Lumpy sizes.** In the worked example the best ratio belongs to a layer of size 2;
  taking it leaves room that the layer of size 3 no longer fits in. The optimum skips
  the best ratio.
- **Non-convex errors.** A layer whose error barely moves from 3 to 4 bits and
  collapses from 4 to 5 has a poor first step and an excellent second one. Greedy
  never looks past the first step.

Greedy is still what most tools do, because the instances are large, the measured
errors are noisy, and a near-optimal answer is good enough. It should be a decision,
not an accident.

## Common mistakes

- **Greedy by error per bit.** See above. It passes every instance where all layers
  have the same size and convex errors, which is what a first test tends to use.
- **An unweighted average.** `mean(bits) <= target` treats a 300-million-parameter
  embedding and a 4-million-parameter projection alike. The file size is
  $\sum n_i b_i$, and the constraint must be on that.
- **Rounding the budget.** `round(target_bpw * N)` admits one bit more than the
  budget whenever the fractional part is at least one half; `<` instead of `<=`
  rejects an assignment that fits exactly. Both are invisible until a budget is
  tight.
- **Rounded costs.** Dividing the budget into a fixed number of buckets and rounding
  each layer's cost to buckets makes the table small for any input, and wrong: the
  rounding errors of several layers add up to more than a bucket, so the result can
  exceed the budget or miss the optimum. Discretize by the gcd, which rounds nothing.
- **Ignoring the tie rules.** With integer errors, ties are common (two identical
  blocks, a layer that does not care). An implementation that keeps the first or the
  last candidate it meets returns a valid optimum but not *the* answer, and its
  output changes when the iteration order does.
- **Returning something when nothing fits.** If even the smallest option is over
  budget, the caller asked for the impossible and must be told.

## Complexity

With $S$ the number of reachable sizes, at most $\text{budget} / g$, the dynamic
program evaluates $O(L \cdot S \cdot K)$ candidates. For 48 layers, 6 options, and
$N / g = 1000$ that is a few hundred thousand, against $6^{48}$ assignments. The time
is pseudo-polynomial: it grows with the numeric value of the budget in units of $g$,
which is why the layer sizes sharing a large divisor matters. For sizes with no
common structure, real tools round the sizes (accepting an approximate budget) or
fall back to greedy.

## In the wild

`mlx_lm.dynamic_quant` (checked against the source of mlx-lm 0.32.2) works with two
options, a low and a high bit width. It estimates one sensitivity per layer: the
gradient of a KL-divergence loss against the unquantized model, taken at the low-bit
weights and projected on the difference between the layer's low-bit and high-bit
weights, divided by the layer's parameter count. It then searches by bisection for a
sensitivity threshold such that giving the high bit width to every layer above it
meets `--target-bpw`. That is the greedy rule: layers are upgraded in order of gain per
parameter until the budget is reached. With two options and a single threshold it is
a reasonable approximation, and it inherits greedy's blind spot for lumpy sizes.

The size model of this problem counts `bits` per weight exactly. Group-wise formats
add a fixed overhead per weight (a scale and a bias per group: half a bit for groups
of 64 with float16 metadata), which changes the cost of an option from `bits` to an
effective figure such as $4.5$; multiplying all costs by a common factor keeps them
integers and leaves the algorithm unchanged.
