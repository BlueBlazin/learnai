Assign each layer of a model its own bit width so the entire model stays within an average bits-per-weight budget while minimizing total quantization error. Each layer has a parameter count and a measured error for every available bit width. You must find the exact globally optimal assignment rather than a greedy approximation.

## Specification

```python
def allocate_bits(layers, options, target_bpw): ...  # -> list of chosen bit widths
```

- `layers`: a list of $L$ dicts in model order. Layer $i$ has the form `{"params": n_i, "errors": [e_i0, e_i1, ...]}`, where $n_i$ is the number of weights in layer $i$ and $e_{ik}$ is its quantization error when assigned `options[k]` bits. Errors are non-negative integers (measured loss increases in fixed-point units), so sums and comparisons are exact. Errors are **not** guaranteed to decrease as bit width increases, and the drop from one option to the next follows no fixed pattern.
- `options`: a list of $K$ distinct positive integers in strictly increasing order representing the available bit widths (for example, `[3, 4, 5, 6, 8]`).
- `target_bpw`: a float giving the maximum allowed average number of bits per weight across the whole model.

An **assignment** selects one bit width $b_i$ from `options` for each layer $i$. Let $k(b_i)$ denote the index of $b_i$ in `options`. The total size in bits and total error of an assignment $b = [b_0, b_1, \dots, b_{L-1}]$ are

$$
\text{bits}(b) = \sum_{i=0}^{L-1} n_i \, b_i, \qquad \text{error}(b) = \sum_{i=0}^{L-1} e_{i,\,k(b_i)}.
$$

Let $N = \sum_{i=0}^{L-1} n_i$ be the total number of weights in the model. An assignment is **feasible** when its parameter-weighted average bit width does not exceed the target:

$$
\frac{\text{bits}(b)}{N} \le \text{target\_bpw}
\quad\Longleftrightarrow\quad
\text{bits}(b) \le \text{target\_bpw} \cdot N.
$$

An assignment whose size equals the budget exactly is feasible.

`allocate_bits` returns the unique feasible assignment that ranks best under these three rules in order:

1. Lowest $\text{error}(b)$.
2. Among ties, lowest $\text{bits}(b)$.
3. Among remaining ties, lexicographically smallest list $[b_0, b_1, \dots, b_{L-1}]$ (compare layer 0 first, then layer 1, and so on).

Return the assignment as a list of $L$ Python ints in layer order. If no assignment is feasible (even choosing the smallest option for every layer exceeds the budget), raise `ValueError`.

The function must return the exact optimum over all $K^L$ assignments without brute-force enumeration, since $L$ can be up to 48.

## Example

```python
layers = [
    {"params": 3, "errors": [30, 0]},
    {"params": 2, "errors": [22, 0]},
    {"params": 1, "errors": [5, 0]},
]
allocate_bits(layers, options=[4, 8], target_bpw=6.0)  # [8, 4, 4]
```

There are $N = 6$ total weights, so the budget is $6.0 \cdot 6 = 36$ bits. Putting every layer at 4 bits uses 24 bits with total error $30 + 22 + 5 = 57$. The remaining 12 bits can be spent in two ways that use the full budget:

| assignment | bits | error |
|---|---|---|
| `[8, 4, 4]` (upgrade layer 0 for 12 bits) | 36 | $0 + 22 + 5 = 27$ |
| `[4, 8, 8]` (upgrade layers 1 and 2 for 8 + 4 bits) | 36 | $30 + 0 + 0 = 30$ |

The optimal assignment is `[8, 4, 4]`. By contrast, a greedy allocator that ranks upgrades by error removed per extra bit picks layer 1 first ($22 / 8 = 2.75$ per bit, ahead of $30 / 12 = 2.5$ for layer 0 and $5 / 4 = 1.25$ for layer 2). Layer 0 then no longer fits in the remaining 4 bits, so greedy upgrades layer 2 instead and finishes with error 30.

For tie-breaking, consider `layers = [{"params": 1, "errors": [9, 0]}, {"params": 1, "errors": [9, 0]}]`, `options = [4, 8]`, and `target_bpw = 6.0`. Both `[4, 8]` and `[8, 4]` have error 9 and use 12 bits. The lexicographically smaller assignment `[4, 8]` wins.

## Constraints

- $1 \le L \le 48$ and $1 \le K \le 6$. Bit widths are integers between 1 and 16.
- $1 \le n_i$ and $N \le 10^{12}$. As in real models, the layer sizes share a large common divisor $g$ such that $N / g \le 1000$. Every achievable size $\text{bits}(b)$ is therefore a multiple of $g$, giving at most $16 \cdot 1000$ distinct sizes.
- `target_bpw` is a multiple of $1/64$ between 1 and 16 (such as `4.5` or `5.015625`), so $\text{target\_bpw} \cdot N$ is exact as a Python float. It does not need to be an integer.
- All errors are integers between 0 and $10^{9}$.
- The returned list is checked for exact equality. On small inputs, it is verified against brute-force enumeration of all $K^L$ assignments.
- A correct solution runs in well under a second on the largest inputs. The time limit per test case is 10 seconds. You may use the Python standard library and NumPy.

## Why this matters

Uniform 4-bit quantization gives the same precision to layers that barely notice rounding and layers that degrade sharply. Mixed-precision tools spend extra bits where sensitivity is highest. For example, `mlx_lm.dynamic_quant` estimates per-layer sensitivity and upgrades sensitive layers to hit a target bits-per-weight budget, and `llama.cpp` GGUF presets mix quantization formats across tensors.

Small implementation mistakes here cause silent failures. Taking an unweighted average over layers lets a massive embedding matrix count the same as a tiny projection, pushing the final checkpoint far over budget. Rounding the bit budget upward can produce a model that overflows the memory limit it was sized for. Relying on a greedy pass leaves budget unused or upgrades the wrong layers whenever layer sizes are uneven, which is the norm in real architectures.
