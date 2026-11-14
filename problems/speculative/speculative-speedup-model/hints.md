Count tokens by asking a different question for each position: with what probability
does the round produce *more than* $i$ tokens? It always produces at least one (the
target's own token). It produces a second one only if the first draft was accepted.
What about the third? Add those probabilities up.

---

The round yields more than $i$ tokens exactly when drafts $1..i$ are all accepted:
probability $\alpha^i$, for $i = 0, \dots, k$. The expectation is the geometric
series $1 + \alpha + \dots + \alpha^k$, with $k + 1$ terms. Its closed form
$(1 - \alpha^{k+1}) / (1 - \alpha)$ is `0 / 0` at $\alpha = 1$, where every term of
the series is 1: handle that value separately (or just sum the terms). For the
speedup, compare with plain decoding in the same currency: plain decoding pays one
target call per token; a round pays $k$ draft calls and one target call for $E$
tokens.

---

`speedup = expected_tokens(alpha, k) / (k * c + 1)`. For `best_k` there is no need
for calculus: evaluate the speedup for every `k` from 0 to `k_max` and keep the
first `k` that attains the largest value. Include `k = 0`, whose speedup is exactly
1; it is the right answer when the draft is too slow or too often wrong.
