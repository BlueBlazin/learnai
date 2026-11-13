Sort descending, take a cumulative sum, and look at which positions have cumulative mass **before** them that is already $\ge p$. Those are the ones to remove.
---
"Cumulative mass before position $r$" is `cumsum - probs`. Comparing that (rather than `cumsum`) against $p$ keeps the token that crosses the threshold.
---
After zeroing the tail, renormalize, take the cumulative sum again, and count how many entries are $\le u$: that count is the sorted rank. Map it back through the sort indices with `gather`.
