Factor the largest term out of the sum before taking the logarithm.
---
$\log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m}$ with $m = \max_i x_i$. The inner sum is between $1$ and $n$.
---
The all-$-\infty$ slice gives $m = -\infty$ and $x_i - m = $ `nan`. Replace a non-finite $m$ with $0$ before subtracting; the final `m + log(sum)` still yields $-\infty$.
