The whole problem is one identity: softmax is shift-invariant, so shift by the maximum.

After the shift every exponent is in $(-\infty, 0]$. The largest term is $e^0 = 1$, which bounds the denominator below by $1$ and above by $V$. Nothing can overflow, and underflow to $0$ in the small terms is the correct rounded answer.

## Why not clamp?

Clamping logits to $[-80, 80]$ also avoids overflow, but it changes the answer: two logits of $900$ and $1000$ would become equal. Shifting is exact.

## Cost

One `max`, one `exp`, one `sum`, one division: $O(V)$ per slice with a single temporary. Production code usually computes `log_softmax` first, as $z - c - \log \sum_j e^{z_j - c}$, because cross-entropy needs the logarithm and exponentiating the result is cheaper than taking a log of a rounded probability.
