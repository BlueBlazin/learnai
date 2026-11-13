Write out the shapes of every intermediate before writing code. What must the shape of $QK^\top$ be, and which axes of `k` have to swap to get it?
---
A boolean mask becomes additive with a `where`: keep the score where the mask is true, otherwise use $-\infty$. Apply it **before** the softmax, not after.
---
`scores = q @ k.transpose(-2, -1) / sqrt(d)`, then mask, then softmax over the last axis, then `weights @ v`. Four lines.
