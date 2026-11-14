Forget tokens for a moment and count bytes. During one decode step for a batch, which
bytes must travel from memory to the processor, and which of them are the same bytes
for every sequence in the batch? Then: if memory can deliver a fixed number of bytes
per second, how many such steps fit in a second, and how many tokens does one step
produce?

---

One step moves `weight_bytes + batch * kv_bytes_read` bytes: the weights once, a
cache per sequence. Convert the bandwidth to bytes per second with `10**9` (not
`2**30`), divide by the bytes per step to get steps per second, and multiply by
`batch`, since every step yields one token for each sequence. For the crossover,
write down the inequality "tokens per second at batch `b` is at least `C`" and solve
it for `b`. Look at the sign of what you divide by: what does it mean when it is zero
or negative?

---

With `B = bandwidth_gb_per_s * 10**9`, `W = weight_bytes`, `K = kv_bytes_read`, and
`C = compute_tokens_per_s`:

$$
\frac{bB}{W + bK} \ge C \iff b\,(B - CK) \ge C\,W .
$$

If `B - C * K <= 0` no batch satisfies it (the bound saturates at `B / K`): return
`None`. Otherwise the answer is the smallest integer at or above
`C * W / (B - C * K)`, which is `math.ceil` of it.
