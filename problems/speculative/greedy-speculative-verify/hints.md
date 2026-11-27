Start by reducing `target_logits` to what greedy decoding cares about, namely one
token index per row. That leaves `K + 1` target tokens and `K` draft tokens. Which
target token pairs with each draft token, and which target token has no partner?

---

Row `i` was computed *before* the target saw `draft_tokens[i]`, so it is the target's
prediction for position `i`. Compare `argmax(target_logits[i])` with `draft_tokens[i]`
for `i` from 0 to `K - 1`. You want the number of matches **before the first
mismatch**, not the total number of matches. Either a loop that breaks on the first
mismatch or a cumulative product of the 0/1 match vector computes this count.

---

Compute `greedy = argmax(target_logits, axis=-1)` of length `K + 1` and
`n = sum(cumprod(greedy[:K] == draft_tokens))`. The next token is `greedy[n]` in both
cases without a branch. When `n < K`, it is the target's correction at the first
mismatch, and when `n == K`, it is the bonus token from row `K`. Convert both values
to Python `int`s before returning.
