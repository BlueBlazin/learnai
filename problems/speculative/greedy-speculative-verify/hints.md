Reduce the logits to what greedy decoding cares about: one token per row. You then
have `K + 1` target tokens and `K` draft tokens. Which target token is compared with
which draft token, and which target token has no partner?

---

Row `i` was computed *before* the target saw `draft_tokens[i]`, so it is the target's
opinion about that very position: compare `argmax(target_logits[i])` with
`draft_tokens[i]` for `i` from 0 to `K - 1`. The number you want is not how many
positions match but how many match **before the first one that does not**. A plain
loop that stops at the first mismatch does it; so does a cumulative product of the
0/1 match vector.

---

With `greedy = argmax(target_logits, axis=-1)` (length `K + 1`) and
`n = sum(cumprod(greedy[:K] == draft_tokens))`, the next token is `greedy[n]` in both
cases: if `n < K` it is the target's token at the first mismatch, and if `n == K` it
is the extra row, the bonus token. No branch is needed. Convert both numbers to
Python ints.
