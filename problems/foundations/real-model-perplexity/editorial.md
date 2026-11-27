## The idea

Perplexity is the exponential of the average negative log-probability per predicted token. When a sequence is longer than the model's context window `max_len`, the only design choice is which context window predicts each token. The strided sliding window uses a simple rule. Windows start every `stride` tokens, and each token is scored by the first window that contains it.

This rule guarantees two properties.

- **Every token is scored once.** Window $k$ ends at $e_k$ and scores positions $e_{k-1}$ through $e_k - 1$ (starting at position 1 for window 0). These ranges partition $1, \dots, N - 1$ with no gaps and no overlap, so the sum always has $N - 1$ terms regardless of `stride`.
- **Every token has context.** Every window before the last has full length `max_len`, so $e_{k-1} = b_k + \texttt{max\_len} - \texttt{stride}$. The first new target of window $k$ therefore sits `max_len - stride` tokens into the window, which is the minimum context guaranteed by the stride. If `stride` were equal to `max_len`, the first new target would be at index 0 of the window and no row would predict it. That is why `stride` must be strictly less than `max_len`.

You only need one integer of state across iterations: `scored`, the first position not yet scored. Inside window $k$ starting at $b_k$, the row that predicts position $p$ is the row for position $p - 1$, which sits at local index $p - 1 - b_k$.

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

The JAX version is identical using `jax.nn.log_softmax`, and the MLX version subtracts `mx.logsumexp`. This is the same loop used in the Hugging Face guide "Perplexity of fixed-length models", which masks out context positions with `-100` labels instead of slicing rows directly.

## Common mistakes

- **Counting the overlap twice.** Scoring every row of every window is a common first attempt. Overlapping tokens then get scored a second time near the start of the next window with almost no context, causing perplexity to rise as overlap increases instead of falling.
- **Disjoint chunks.** Advancing by `max_len` and ignoring `stride` drops the first token of every chunk (since no row predicts it) and scores the next few tokens with almost no context. On the real model, this degrades perplexity by tens of percent.
- **Dropping the short last window.** Looping only over full windows skips the tail of the sequence. Dividing by the number of tokens actually scored hides the missing tail.
- **Averaging window means.** The first window scores `max_len - 1` tokens, middle windows score `stride` tokens, and the last window scores whatever remains. Averaging per-window means weights tokens unequally. Sum the log-probabilities over all tokens and divide once at the end.
- **Dividing by `N`.** Token 0 is never predicted, so there are `N - 1` terms in the sum.
- **Unshifted targets.** Reading token `p` from local row `p - begin` scores each token using logits computed *after* the model already saw that token. On a real model, the resulting number is not obviously broken, which makes the bug easy to miss.
- **Bits versus nats.** Base-2 logarithms (`log2`) pair with `2 ** (-mean)`, whereas `exp` requires the natural logarithm. Mixing them raises the perplexity to the power $1 / \ln 2$.
- **One call on the whole text.** Passing all of `token_ids` at once works only when $N \le$ `max_len`. Longer sequences raise `ValueError`, matching a real model's context limit.
- **Adding an epsilon inside `log`.** Computing `log(softmax(x) + 1e-8)` clamps every log-probability at $-18.4$. Toy models rarely hit that floor, but Granite assigns a made-up word in one hidden test a probability near $e^{-36}$, where clamping shifts the perplexity from 125 to 62.

## Complexity

The loop makes $\lceil \max(N - \texttt{max\_len},\, 0) / \texttt{stride} \rceil + 1$ calls to `forward`, each processing at most `max_len` tokens. In total, the model processes roughly $N \cdot \texttt{max\_len} / \texttt{stride}$ tokens. Halving the stride doubles the evaluation cost, and a stride of 1 multiplies it by `max_len`. The bookkeeping around the calls is $O(N)$.

## In the wild

Published perplexities are only comparable when both the window length and the stride are reported. For the same model, perplexity can shift by several percent between a stride of half the window and a stride equal to the full window. `llama.cpp`'s `perplexity` tool scores only the second half of each chunk for the same reason, using the first half solely as context. When a model's context window is long enough to hold the sequence, a KV cache avoids recomputing the overlap. The strided window is what you use when the text exceeds that context limit.
