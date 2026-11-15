Suppose you have already decided the bit widths of the first $i$ layers. What do the
remaining layers need to know about those decisions? Not which layer got what, only
two numbers: how many bits have been used, and how much error has accumulated. And if
two partial assignments have used the same number of bits, is there any reason to
keep the one with the higher error?

---

Dynamic programming over the bits used. Keep a table that maps "bits used by the
layers so far" to the best (lowest-error) partial assignment with exactly that size.
Start with `{0: (0, ())}`. For each layer, build a new table: for every entry and
every option, add the layer's cost `params * bits` and its error; drop the candidate
if it exceeds the budget; keep it if its size is new or it beats the entry already
stored for that size. The budget is `floor(target_bpw * total_params)` bits: a size is
a whole number of bits.

A dict keyed by the size is enough. All sizes are multiples of the gcd of the layer
sizes, so the table has at most `budget / gcd` entries however large the layers are.

---

Store each entry as a tuple `(error, chosen_bits_so_far)` and compare tuples: Python
compares the error first and then the chosen bit widths in layer order, which is the
lexicographic tie rule for two candidates of the same size.

```python
candidate = (error + layer_error, (*chosen, bits))
if total not in extended or candidate < extended[total]:
    extended[total] = candidate
```

At the end, take `min((error, used, chosen) for used, (error, chosen) in best.items())`:
lowest error, then fewest bits, then the smallest list. If the table is empty after
some layer, nothing fits: raise `ValueError`.
