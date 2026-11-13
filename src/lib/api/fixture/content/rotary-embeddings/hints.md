The angle table is an outer product: positions $(T,)$ times frequencies $(d/2,)$ gives $(T, d/2)$.
---
Split `x` into even and odd components with `x[..., 0::2]` and `x[..., 1::2]`. The rotation is two multiply-adds on those halves.
---
`even' = even * cos - odd * sin` and `odd' = even * sin + odd * cos`. Interleave them back with a `stack` on a new last axis followed by a `flatten`.
