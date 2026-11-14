## The idea

Perplexity is $\exp$ of the average negative log-probability per predicted token. The
only freedom is in *which context* each token is predicted from, and a model with a
window of `max_len` tokens forces a choice. The strided window makes one that is easy
to state: windows start every `stride` tokens, and a token belongs to the first window
that reaches it.

That rule has two consequences worth seeing once.

- **Every token is scored once.** Window $k$ ends at $e_k$, and it scores exactly the
  positions from $e_{k-1}$ to $e_k - 1$. The ranges tile $1, \dots, N - 1$ with no gap
  and no overlap, so the sum has $N - 1$ terms whatever the stride.
- **Every token has context.** All windows but the last are full, so
  $e_{k-1} = b_k + \texttt{max\_len} - \texttt{stride}$: the first new target of window
  $k$ sits `max_len - stride` tokens into the window. That is the context the stride
  buys. With `stride = max_len` the first new target would be the first token of the
  window, which no row predicts, and that is why the stride must be smaller.

One integer of state, the position up to which the text is scored, is enough. The row
arithmetic is the shift of `sequence-log-likelihood` inside a window: the row that
predicts position $p$ is the row of position $p - 1$, at index $p - 1 - b_k$.

## Reference solution (PyTorch)

```python
def perplexity(forward, token_ids, max_len, stride):
    n = token_ids.shape[0]
    total = torch.zeros((), dtype=torch.float32)
    scored = 1  # token 0 has no context to be scored from
    for begin in range(0, n, stride):
        end = min(begin + max_len, n)
        logp = torch.log_softmax(forward(token_ids[begin:end]), dim=-1)
        rows = torch.arange(scored - 1 - begin, end - 1 - begin)
        total = total + logp[rows, token_ids[scored:end]].sum()
        scored = end
        if end == n:
            break
    return torch.exp(-total / (n - 1))
```

The JAX version is the same with `jax.nn.log_softmax`; the MLX version subtracts
`mx.logsumexp`. This is the loop of the Hugging Face guide "Perplexity of fixed-length
models", which hides the row selection behind labels of `-100`.

## Common mistakes

- **Counting the overlap twice.** Scoring every row of every window is the natural
  first version. Tokens in an overlap are then counted twice, the second time from the
  start of a window with almost no context, and the perplexity rises with the amount
  of overlap, the opposite of what a smaller stride should do.
- **Disjoint chunks.** Advancing by `max_len` and ignoring the stride loses the first
  token of every chunk (no row predicts it) and scores the next few nearly blind. On
  the real model this costs tens of percent.
- **Dropping the short last window.** A loop that only runs full windows never scores
  the tail of the text. Dividing by the number of tokens it did score hides it.
- **Averaging window means.** The first window scores `max_len - 1` tokens, the middle
  ones `stride`, the last one whatever is left. A mean of per-window means weights
  tokens unequally. Sum over tokens, divide once.
- **Dividing by `N`.** There are `N - 1` predictions.
- **Unshifted targets.** Reading token `p` from row `p - b` scores a token under
  logits that were computed after reading it. With a real model the result is not
  obviously wrong, which is what makes it dangerous.
- **Bits versus nats.** `log2` probabilities belong with `2 ** mean`; `exp` needs the
  natural logarithm. Mixing them raises the perplexity to the power $1 / \ln 2$.
- **One call on the whole text.** Correct for a text that fits, impossible otherwise:
  `forward` refuses more than `max_len` tokens, as a real context window does.
- **An epsilon in the logarithm.** `log(softmax(x) + 1e-8)` clamps every
  log-probability at $-18.4$. A toy model never gets near that; Granite gives the
  made-up word in one of the hidden passages a probability of $e^{-36}$, and the
  clamp changes the perplexity from 125 to 62.

## Complexity

$\lceil (N - \texttt{max\_len}) / \texttt{stride} \rceil + 1$ model calls of at most
`max_len` tokens each, so the model reads about $N \cdot \texttt{max\_len} /
\texttt{stride}$ tokens: a stride of half the window doubles the cost of the
evaluation, and a stride of 1 multiplies it by `max_len`. The bookkeeping is $O(N)$.

## In the wild

Published perplexities are only comparable when window and stride are stated; the
numbers for the same model move by several percent between a stride of half the window
and a stride of the whole window. `llama.cpp`'s `perplexity` tool scores only the
second half of each chunk for the same reason: the first half is context. With a KV
cache the overlap need not be recomputed at all for a model whose context is long
enough; the strided window is what you do when it is not.
