Start by counting bytes rather than tokens. During one decode step for a batch, which bytes must move from memory to the processor, and which of those bytes are shared across every sequence in the batch? Once you have bytes per step, use the memory bandwidth to find how many steps fit in one second, and multiply by how many tokens each step produces.

---

One step transfers `weight_bytes + batch * kv_bytes_read` bytes, reading the weights once and one KV cache per sequence. Convert `bandwidth_gb_per_s` to bytes per second using `10**9` (not `2**30`), divide by the bytes per step to get steps per second, and multiply by `batch` because each step emits one token per sequence. For `crossover_batch`, write the inequality $M(b) \ge C$ and solve for `b`. Check the sign of the coefficient you divide by, and consider what a zero or negative value means.

---

Let `B = bandwidth_gb_per_s * 10**9`, `W = weight_bytes`, `K = kv_bytes_read`, and `C = compute_tokens_per_s`.

$$
\frac{bB}{W + bK} \ge C \iff b\,(B - CK) \ge C\,W .
$$

If `B - C * K <= 0`, no batch can reach `C` because throughput saturates at `B / K`, so return `None`. Otherwise, return `math.ceil(C * W / (B - C * K))`.
