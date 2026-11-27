For each entry of the draft, note three things: which position it predicts, which
hidden state it reads, and which matrix projects that state to logits. For
`draft[0]`, those are position $t + 1$, `hidden`, and `w_out`. For `draft[j]` with
$j \ge 1$, only two of those three change. Identify which one stays the same across
every head, and check whether a loop that overwrites a `state` variable would break
that rule.

---

Every head reads the same input `hidden`. For head `(w1, b1, w2)`, compute
`z = w1 @ hidden + b1`, then `state = hidden + z * sigmoid(z)`, and take
`argmax(w2 @ state)`. Notice that the residual adds the *activated* branch to the
untouched `hidden` vector rather than applying SiLU after the addition. Iterate over
`heads[:k]` in order, with the model's own token `argmax(w_out @ hidden)` at index 0.

---

For `mtp_accept`, align the two vectors by the sequence position they predict.
`target_argmax[j - 1]` is the model's greedy token for the position immediately
after `draft[j - 1]`, which is the position `draft[j]` guessed. Compare `draft[1:]`
with `target_argmax[:-1]`, count the length of the leading run of `True` values
(for example, by summing the cumulative product), and add 1 for `draft[0]`, which
is always accepted.
