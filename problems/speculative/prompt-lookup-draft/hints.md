Fix one n-gram length $n$ first. The suffix is `context[L - n :]`, and each window `context[s : s + n]` is a candidate. Which values of $s$ are valid, and which single start index must be excluded so you do not match the suffix with itself?

---

Valid start indices are $s = 0, \dots, L - n - 1$. Scan $s$ upward from 0 and stop at the first match to guarantee the earliest occurrence. Wrap this scan in an outer loop over $n$ from `max_ngram` down to `min_ngram`, returning immediately on the first hit because a longer n-gram always beats a shorter one even if the shorter match appears earlier.

---

The draft tokens start immediately after the matched window at `context[s + n : s + n + k]`. Python slicing automatically stops at the end of the list, which handles drafts shorter than `k`. Start the outer loop at `min(max_ngram, L - 1)` so short contexts need no special case, and return `[]` if both loops finish without a match.
