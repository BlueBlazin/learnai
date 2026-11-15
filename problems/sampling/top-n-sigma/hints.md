The rule is one comparison per token: `logit >= max - n * std`. The maximum is easy.
For the standard deviation, two questions decide everything: what happens to a
library `std` when one entry of the row is `-inf`, and which divisor does your
framework's `std` use by default?

---

Compute the statistics yourself, over the finite entries only. Build a boolean table
`finite = isfinite(logits)`, count the finite entries of each row (keep the reduced
axis, so the count has shape `(B, 1)`), and take sums in which masked entries are
replaced by 0. The mean is that sum divided by the count; the variance is the sum of
squared deviations of the finite entries divided by the same count. Handle `n <= 0`
first by returning the logits.

---

```
finite = isfinite(logits)
count = sum(finite, axis=-1, keepdims=True)
mean = sum(where(finite, logits, 0), axis=-1, keepdims=True) / count
var = sum(where(finite, (logits - mean) ** 2, 0), axis=-1, keepdims=True) / count
cutoff = max(logits, axis=-1, keepdims=True) - n * sqrt(var)
out = where(logits >= cutoff, logits, -inf)
```

`(logits - mean) ** 2` is `inf` at a masked entry, but `where` selects the 0 there, so
the infinity never enters a sum. With `>=`, a row whose finite logits are all equal
has `var = 0`, a cutoff equal to its maximum, and keeps all of them; a masked entry
fails `-inf >= cutoff` and stays `-inf` without a special case.
