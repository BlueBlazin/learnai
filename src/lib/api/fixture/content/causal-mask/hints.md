A lower-triangular matrix of ones is the causal part. What is the padding part, as a comparison between a range and `lengths`?
---
`arange(max_len) < lengths[:, None]` gives a $(B, T)$ key-validity mask. Broadcast it across the query axis and AND it with the triangle.
