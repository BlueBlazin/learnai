Fix one n-gram length $n$ first. The suffix is `context[L-n:]`. Every window
`context[s:s+n]` is a candidate; which values of $s$ are allowed, and which single
value must be excluded because it would compare the suffix with itself?

---

Allowed starts are $s = 0, \dots, L - n - 1$. You want the smallest $s$ whose window
equals the suffix, so scan $s$ upward from 0 and stop at the first hit.
Wrap that in a loop over $n$ from `max_ngram` down to `min_ngram`, and return as
soon as any $n$ produces a hit: a longer match always beats a shorter one, even if
the shorter one occurs earlier.

---

The tokens to copy start right after the window: `context[s+n : s+n+k]`. Python
slicing already stops at the end of the list, which is the "fewer than `k` tokens"
rule. Start the outer loop at `min(max_ngram, L - 1)` so that short contexts need no
special case, and return `[]` after both loops finish without a hit.
