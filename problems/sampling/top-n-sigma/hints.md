The rule is a single comparison per token: `logit >= max - n * std`. Computing the row maximum is straightforward. For the standard deviation, check two things first. What does your framework's built-in `std` return when a row contains `-inf`, and does it divide by the count or by count minus one by default?

---

Compute the mean and variance directly over the finite entries. Build a boolean mask `finite = isfinite(logits)` and count the finite entries per row with `keepdims=True` (giving shape `(B, 1)`). Replace non-finite entries with `0` before summing to get the row mean, then replace non-finite squared deviations with `0` before summing and dividing by the same finite count to get the variance. Handle `n <= 0` up front by returning `logits` immediately.

---

```
finite = isfinite(logits)
count = sum(finite, axis=-1, keepdims=True)
mean = sum(where(finite, logits, 0), axis=-1, keepdims=True) / count
var = sum(where(finite, (logits - mean) ** 2, 0), axis=-1, keepdims=True) / count
cutoff = max(logits, axis=-1, keepdims=True) - n * sqrt(var)
out = where(logits >= cutoff, logits, -inf)
```

At a masked entry, `(logits - mean) ** 2` produces `inf`, but `where` selects `0` so no infinity enters the sum. Using `>=` handles zero-variance rows cleanly because `var` is `0`, the cutoff equals the maximum logit, and all tied maximum tokens pass. Any masked `-inf` entry fails `-inf >= cutoff` and remains `-inf` without a special case.
