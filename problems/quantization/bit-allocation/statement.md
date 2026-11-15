Give every layer of a model its own bit width so that the model fits a size budget
with the least total error. Each layer comes with its number of parameters and a
measured error for each available bit width; you choose one width per layer. The
natural greedy rule, upgrade whatever buys the most error per bit, is not optimal,
and the answer has to be: this is a knapsack problem, solved exactly.

## Specification

```python
def allocate_bits(layers, options, target_bpw): ...  # -> list of chosen bit widths
```

- `layers`: a list of $L$ dicts, one per layer, in model order. Layer $i$ is
  `{"params": n_i, "errors": [e_i0, e_i1, ...]}`: its number of weights $n_i$, and
  its error $e_{ik}$ if it is quantized to `options[k]` bits. Errors are
  non-negative integers (a measured loss increase in fixed-point units), so sums and
  comparisons of errors are exact. They are **not** guaranteed to decrease with the
  bit width, and the decrease from one option to the next follows no pattern.
- `options`: a list of $K$ distinct positive integers in increasing order, the
  available bit widths, for example `[3, 4, 5, 6, 8]`.
- `target_bpw`: a float, the largest allowed average number of bits per weight.

An **assignment** chooses one bit width $b_i$ from `options` for every layer. Its size
and its error are

$$
\text{bits}(b) = \sum_{i} n_i \, b_i, \qquad \text{error}(b) = \sum_{i} e_{i,\,k(b_i)},
$$

where $k(b_i)$ is the position of $b_i$ in `options`. With $N = \sum_i n_i$, an
assignment is **feasible** when its average is at most the target, weighted by layer
size:

$$
\frac{\text{bits}(b)}{N} \le \text{target\_bpw}
\quad\Longleftrightarrow\quad
\text{bits}(b) \le \text{target\_bpw} \cdot N .
$$

A size exactly equal to the budget is feasible.

`allocate_bits` returns the feasible assignment that is best in this order:

1. the lowest $\text{error}(b)$;
2. among those, the lowest $\text{bits}(b)$;
3. among those, the lexicographically smallest list $[b_0, b_1, \dots]$ (compare the
   bit widths of layer 0; if equal, of layer 1; and so on).

These three rules make the answer unique. It is returned as a list of $L$ Python
ints, the bit width of each layer in order.

If no assignment is feasible (even the smallest option everywhere exceeds the
budget), raise `ValueError`.

The result must be the true optimum over all $K^L$ assignments, found without
enumerating them: $L$ is up to 48.

## Example

```python
layers = [
    {"params": 3, "errors": [30, 0]},
    {"params": 2, "errors": [22, 0]},
    {"params": 1, "errors": [5, 0]},
]
allocate_bits(layers, options=[4, 8], target_bpw=6.0)  # [8, 4, 4]
```

There are $N = 6$ weights, so the budget is $6.0 \cdot 6 = 36$ bits. Everything at 4
bits costs 24 bits and has error $30 + 22 + 5 = 57$. The remaining 12 bits can be
spent in two ways that use all of them:

| assignment | bits | error |
|---|---|---|
| `[8, 4, 4]`: upgrade layer 0 (12 bits) | 36 | $0 + 22 + 5 = 27$ |
| `[4, 8, 8]`: upgrade layers 1 and 2 (8 + 4 bits) | 36 | $30 + 0 + 0 = 30$ |

The answer is `[8, 4, 4]`. A greedy allocator that ranks upgrades by error removed
per bit starts with layer 1 ($22 / 8 = 2.75$ per bit, against $30 / 12 = 2.5$ for
layer 0 and $5 / 4 = 1.25$ for layer 2). After that, layer 0 no longer fits, so it
takes layer 2 and ends at error 30.

Ties: with `layers = [{"params": 1, "errors": [9, 0]}, {"params": 1, "errors": [9,
0]}]`, `options = [4, 8]`, and `target_bpw = 6.0`, the assignments `[4, 8]` and
`[8, 4]` both have error 9 and 12 bits; the lexicographically smaller one, `[4, 8]`,
is the answer.

## Constraints

- $1 \le L \le 48$ and $1 \le K \le 6$; bit widths are between 1 and 16.
- $1 \le n_i$ and $N \le 10^{12}$. The layer sizes share a large common divisor $g$,
  as the layers of a real model do: $N / g \le 1000$. Every size $\text{bits}(b)$ is
  therefore a multiple of $g$, and there are at most $16 \cdot 1000$ distinct sizes.
- `target_bpw` is a multiple of $1/64$ between 1 and 16 (for example `4.5` or
  `5.015625`), so the product $\text{target\_bpw} \cdot N$ is exact in a Python
  float. It need not be a whole number of bits.
- Errors are integers between 0 and $10^{9}$.
- The returned list is compared **exactly**. On small inputs it is compared with
  the result of enumerating every assignment.
- A correct solution takes well under a second on the largest input; the time limit
  per case is 10 seconds. You may use the standard library and numpy.

## Why this matters

Uniform 4-bit quantization treats a layer that barely notices rounding and a layer
that falls apart at 4 bits the same way. Mixed-precision recipes spend the bits where
they matter: `mlx_lm.dynamic_quant` estimates a sensitivity for every layer and gives
the sensitive ones more bits until a target number of bits per weight is reached, and
the GGUF presets of llama.cpp mix quantization types across tensors. Underneath is
this optimization problem. Its details decide whether the result is what was asked
for. The average must be weighted by layer size, or an embedding matrix with a third
of the parameters counts as one layer among a hundred and the file comes out far over
budget. The budget is a hard limit in bits, so a rounding that lets the model exceed
it by one layer's upgrade is a model that does not fit in the memory it was sized
for. And a greedy pass can leave budget unused, or spend it on the wrong layer, in
exactly the cases where the layer sizes are lumpy, which they are.
