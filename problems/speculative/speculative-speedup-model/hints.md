Count the expected tokens by asking, for each position $i$, what the probability is that a round produces *more than* $i$ tokens. Every round produces at least 1 token (the target's own token), and it produces a second token only if the first draft is accepted. What is the probability for the third token and beyond? Sum those probabilities.

---

A round yields more than $i$ tokens exactly when the first $i$ drafts are all accepted, which has probability $\alpha^i$ for $i = 0, \dots, k$. The expectation is therefore the geometric series $1 + \alpha + \dots + \alpha^k$ with $k + 1$ terms. Its closed form $(1 - \alpha^{k+1}) / (1 - \alpha)$ evaluates to `0 / 0` at $\alpha = 1$, where every term is 1, so handle `alpha == 1.0` separately (or sum the terms directly). For `speedup`, compare both methods in target-call units: plain decoding pays 1 target call per token, while a speculative round pays $k$ draft calls and 1 target call for $E$ tokens.

---

Compute `speedup = expected_tokens(alpha, k) / (k * c + 1)`. For `best_k`, you do not need calculus. Evaluate `speedup` for every integer `k` from `0` to `k_max` and return the first `k` that attains the maximum. Make sure to include `k = 0` (whose speedup is `1.0`), because it is the right answer whenever speculation slows decoding down.
