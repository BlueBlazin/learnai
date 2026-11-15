There are five operations in total: a lookup, a multiplication, a matrix product, a
division, and a `tanh`. The problem is entirely in where each constant goes and in
what order. For each constant, ask: which tensor does it act on, and is it a factor or
a divisor?

---

`embed`: index the table with the token tensor (`table[tokens]` works in all three
frameworks and keeps the shape of `tokens` in front), then multiply the looked-up rows
by `scale`. Multiply the rows, not the table: `table *= scale` changes the matrix for
whoever else holds it, and with tied embeddings that is the output head.

`finalize_logits`: `w_out` is `(V, d)`, so the product is `h @ w_out.T`. Divide by
`logit_scale`. Then, only if `softcap` is not `None`, apply the cap to what you have
at that point.

---

```python
logits = h @ w_out.T / logit_scale
if softcap is not None:
    logits = softcap * tanh(logits / softcap)
```

The cap divides by `softcap` inside the `tanh` and multiplies by it outside, so that
small logits come out unchanged ($\tanh u \approx u$) and large ones approach
$\pm$`softcap`. Test for `None` explicitly; do not write `if softcap:`.
