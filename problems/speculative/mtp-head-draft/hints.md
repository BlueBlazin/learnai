Write down, for each entry of the draft, three things: which position it is a
guess for, which state it is computed from, and which matrix turns that state into
logits. For `draft[0]` the answers are $t + 1$, `hidden`, `w_out`. For `draft[j]`
with $j \ge 1$ only two of the three change. Which one stays the same for every
head, and what does that say about a loop that updates a `state` variable from one
head to the next?

---

Every head starts from `hidden`. For head `(w1, b1, w2)`: `z = w1 @ hidden + b1`,
`state = hidden + z * sigmoid(z)`, token `argmax(w2 @ state)`. The residual adds
the *activated branch* to the untouched hidden state; the activation is not
applied to the sum. Use `heads[:k]`, in order, and put the model's own token
`argmax(w_out @ hidden)` in front.

---

For `mtp_accept`, line the two vectors up by the position they talk about.
`target_argmax[j - 1]` is the model's token for the position after
`draft[j - 1]`, which is the position `draft[j]` guessed. So compare `draft[1:]`
with `target_argmax[:-1]`, keep the leading run of matches (a cumulative product,
summed), and add 1 for `draft[0]`, which needs no check.
