Suppose you have already chosen bit widths for the first $i$ layers. The remaining layers do not need to know which specific widths earlier layers received. They only need two numbers: how many bits have been used so far and how much error has accumulated. If two partial assignments have used the same number of bits, can the one with higher error ever beat the other?

---

Use dynamic programming over total bits used. Maintain a table mapping each reachable bit count from the layers processed so far to the lowest-error partial assignment of that size. Start with `{0: (0, ())}`. For each layer, build a new table by combining every existing entry with every option: add `params * bits` to the size and `layer_error` to the error, discard candidates above the budget, and keep a candidate if its size is new or beats the stored entry. Because every valid size is a whole number of bits, the budget is `floor(target_bpw * total_params)`.

A dictionary keyed by total bits is fast enough without explicit scaling. Every reachable size is a multiple of the gcd of the layer sizes, so the table holds at most `budget / gcd` entries regardless of how large the parameter counts are.

---

Store each table value as a tuple `(error, chosen_bits_so_far)`. Comparing two tuples in Python compares error first and then the chosen bit widths in layer order, which handles the lexicographic tie-breaker automatically:

```python
candidate = (error + layer_error, (*chosen, bits))
if total not in extended or candidate < extended[total]:
    extended[total] = candidate
```

After processing all layers, raise `ValueError` if the table is empty. Otherwise, take `min((error, used, chosen) for used, (error, chosen) in best.items())` to pick the lowest error, then fewest bits, then lexicographically smallest list.
