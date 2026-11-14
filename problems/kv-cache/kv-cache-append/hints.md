The score matrix of one step is not square: it has one row per **new** query and one
column per entry of the cache after appending, so its shape is `(T, S + T)`. Take
`S = 3` and `T = 2` and write down, for each of the two rows, which of the five
columns the query may see. What is the absolute position of new token `i`?

---

Concatenate first, on axis 2, cached entries before new ones (and skip the
concatenation when `cache` is `None`). Then the column index `j` of the score matrix
is simply the absolute position of the entry, and new token `i` is the token at
absolute position `S + i`. The causal rule "a token sees itself and the past" is the
same as ever, written in absolute positions: `j <= S + i`. With `S = 0` it is the
lower triangle you know; with `T = 1` every column is visible.

---

Build the boolean mask of shape `(T, S + T)` from two `arange`s that broadcast against
each other:

```
visible[i, j] = (j <= S + i)        # arange(S + T)[None, :] <= S + arange(T)[:, None]
```

and pass it to `ops.attention(q_new, K, V, visible)`, where True means "may attend";
it broadcasts over the batch and head axes. Return the output together with the
concatenated `(K, V)`, which is the cache for the next call.
