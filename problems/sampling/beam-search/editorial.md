## The idea

Greedy decoding commits to the single best token at every step. Beam search instead tracks the `n_beams` most probable prefixes, allowing a path that starts with a slightly less likely token to win later. The algorithm rests on three pieces.

**One candidate table per step.** Let `V` be the vocabulary size and `S` be the 1D array of cumulative log-probabilities for the current `rows` live beams. Computing `S[:, None] + log_softmax(logits)` forms a `(rows, V)` table of candidate log-probabilities. Flattening this table maps candidate `(row, token)` to index `row * V + token`, so selecting the top candidates across all beams reduces to a single sort. Candidates must be scored with log-probabilities rather than raw logits, because raw logits are only defined up to an arbitrary per-row shift and cannot be summed across different beams.

**Keeping the cache aligned with the beams.** After selecting the next live beams, new beam `j` extends old row `parents[j]`. Multiple new beams often share a parent, while some old rows have no children. When rebuilding the token lists from `parents`, we must reorder the cache with `cache.select(parents)` before feeding the new tokens. This preserves the core invariant: *row `j` of the cache holds exactly the prompt followed by the tokens of live beam `j`.* After step 1, `cache.select` also broadcasts the single prompt row to `n_beams` rows, so the prompt is only processed once.

**Finishing and early stopping.** A candidate ending in EOS leaves the live beams and receives the length-normalized score $S / L^{\alpha}$, where $L$ is its generated length and $\alpha$ is `length_penalty`. With $\alpha = 0$, every extra token lowers the score and short outputs dominate. With $\alpha = 1$, hypotheses are compared by average log-probability per token. Normalization applies only to finished hypotheses. Live beams within a step always share the same length, so normalizing does not change their ranking, and carrying a normalized value into the next step would corrupt the sum by adding raw log-probabilities to an average.

Two rules keep pruning consistent. First, only candidates ranked below `n_beams` may finish, because an EOS at rank `n_beams` or worse would have been pruned. Second, taking `2 * n_beams` candidates guarantees `n_beams` live beams survive every step, since each live beam produces at most one EOS candidate.

The stopping rule uses an exact upper bound. Let $S_{\text{best}}$ be the highest cumulative log-probability among the live beams. Every future descendant of a live beam has cumulative log-probability $S' \le S_{\text{best}}$ (each new token adds a value $\le 0$) and length $L \le \texttt{max\_new\_tokens}$. Since $S' \le 0$ and $\alpha \ge 0$,

$$
\frac{S'}{L^{\alpha}} \;\le\; \frac{S'}{\texttt{max\_new\_tokens}^{\alpha}}
\;\le\; \frac{S_{\text{best}}}{\texttt{max\_new\_tokens}^{\alpha}} .
$$

Once this upper bound is at most the worst score in a full finished set, no live beam can enter the finished set, so stopping early leaves the result unchanged while saving model calls.

## Reference solution (PyTorch)

```python
def beam_search(lm, prompt, n_beams, max_new_tokens, eos_id, length_penalty):
    logits, cache = lm(prompt)
    log_probs = ops.log_softmax(logits[-1:])  # (1, V)
    beams, scores, finished = [[]], [0.0], []

    for step in range(1, max_new_tokens + 1):
        vocab = log_probs.shape[-1]
        totals = (torch.tensor(scores)[:, None] + log_probs).reshape(-1)
        order = torch.argsort(totals, descending=True, stable=True)[: 2 * n_beams]

        live = []  # (parent row, token, log-probability)
        for rank, (index, total) in enumerate(zip(order.tolist(), totals[order].tolist())):
            parent, token = divmod(index, vocab)
            if step == max_new_tokens or token == eos_id:
                if rank < n_beams:
                    finished.append((total / step**length_penalty, [*beams[parent], token]))
            elif len(live) < n_beams:
                live.append((parent, token, total))

        finished.sort(key=lambda hypothesis: -hypothesis[0])
        del finished[n_beams:]
        if step == max_new_tokens:
            break
        bound = live[0][2] / max_new_tokens**length_penalty
        if len(finished) == n_beams and bound <= finished[-1][0]:
            break

        parents = [parent for parent, _, _ in live]
        beams = [[*beams[parent], token] for parent, token, _ in live]
        scores = [total for _, _, total in live]
        logits, cache = lm([[token] for _, token, _ in live], cache.select(parents))
        log_probs = ops.log_softmax(logits[:, -1, :])

    return [(tokens, score) for score, tokens in finished]
```

Step 1 requires no special case because we start with one empty live beam and a single-row `log_probs` tensor. Each step performs one `log_softmax` and one sort on tensors, followed by a tiny Python loop over `2 * n_beams` candidates. Python's `list.sort` is stable, so ties automatically keep the earlier-added hypothesis ahead.

## Common mistakes

- **Not reordering the cache.** Expanding the cache once and feeding new tokens row by row only works while new beam `j` happens to descend from old row `j`. As soon as beams reshuffle, each row silently continues the wrong history without raising an error.
- **Letting finished hypotheses keep expanding.** Keeping an EOS candidate in the live list wastes a beam slot and asks the model to generate tokens after EOS. Implementations avoid this either by removing finished beams (as above) or by setting their continuation scores to $-\infty$.
- **Normalizing before pruning.** Storing $S / L^{\alpha}$ as a live beam's running score causes the next step to add a raw log-probability to a length-normalized average.
- **Taking only `n_beams` candidates.** Every EOS in the top `n_beams` then shrinks the number of live beams, collapsing the search toward greedy decoding right after the first hypothesis finishes.
- **Finishing every EOS in the top `2 * n_beams`.** The extra `n_beams` candidates exist only to refill live beams. An EOS at rank `n_beams` or worse was pruned and must not finish.
- **Counting the prompt in the length.** Because $S$ sums only over generated tokens, $L$ must exclude the prompt length $P$. For a long prompt, $(P + L)^{\alpha}$ barely varies with $L$, silently neutralizing the length penalty. (Hugging Face transformers subtracts `decoder_prompt_len` for this reason.)
- **Using raw logits as scores.** Sums of unnormalized logits are not comparable across beams.
- **Stopping as soon as the finished set is full.** The first `n_beams` hypotheses to finish are not always the highest-scoring ones. In the second example of the statement, the final runner-up finishes two steps later.
- **Bounding with the current length.** When $\alpha > 0$, $S_{\text{best}} / t^{\alpha}$ is not a valid upper bound because longer hypotheses divide by a larger denominator. Using $t$ instead of `max_new_tokens` stops the second example prematurely at step 3.
- **Never stopping early, or re-feeding prefixes.** Both produce the right hypotheses but fail the cost limits by making extra model calls or processing $O(\text{steps}^2)$ tokens.

## Complexity

Let $P$ be the prompt length and $s$ be the number of steps taken. Beam search makes $s$ model calls and processes $P + n_{\text{beams}} (s - 1)$ tokens, compared to $P + s - 1$ for greedy decoding. Decoding work therefore scales roughly by `n_beams`, executed in batches of size `n_beams`. Each step sorts $n_{\text{beams}} \cdot V$ scores (or selects the top $2 n_{\text{beams}}$ in linear time). Calling `cache.select` copies $O(n_{\text{beams}} \cdot \text{length})$ cache entries per layer per step. In production inference engines, this cache copying is the main bottleneck and motivates storing beams as a tree of shared cache blocks.

Exact stopping also interacts with $\alpha > 0$. Even with `n_beams = 1` and $\alpha = 1$, the search cannot stop at the first EOS because a longer hypothesis could still achieve a higher average log-probability. Exact optimality and length normalization pull in opposite directions.

## In the wild

This specification matches `_beam_search` in Hugging Face transformers (`generation/utils.py`). It takes `2 * num_beams` candidates per step, allows only the top `num_beams` to finish, scores finished hypotheses as `log_prob / length ** length_penalty` (excluding prompt length), and retains the top `num_beams` finished hypotheses. The stopping rule here corresponds to `early_stopping="never"`, which the source treats as the canonical algorithm. Its default `early_stopping=False` uses the current length in the bound as a heuristic for backward compatibility (the "bounding with the current length" pitfall above), while `early_stopping=True` stops as soon as `num_beams` hypotheses finish. Its cache reordering uses the same row-gather operation as `cache.select` across every layer's keys and values.

Beam search is uncommon in general-purpose LLM serving engines. vLLM implements it in an outer entrypoint (`vllm/entrypoints/generate/beam_search`) on top of regular generation requests rather than inside the core engine, and llama.cpp's public API omits it. Sampling dominates open-ended text generation because maximum-probability sequences tend to be repetitive. Beam search remains standard for tasks with a narrow set of correct outputs, including speech recognition (Whisper's CLI defaults to beam size 5), machine translation, and constrained decoding.
