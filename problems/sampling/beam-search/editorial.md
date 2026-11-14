## The idea

Greedy decoding commits to the best token at every step. Beam search keeps the
`n_beams` best *prefixes* instead, so a path that starts with a slightly less likely
token can still win. Three pieces make up the algorithm.

**One table per step.** With the live beams as rows, the log-probabilities of all
candidates are `S[:, None] + log_softmax(logits)`, a `(rows, V)` table. Flattened,
candidate `row * V + token` is one number, and "the best candidates over all beams"
is a single sort. Scores must be log-probabilities: raw logits are only defined up
to a constant per row, so sums of logits from different beams cannot be compared.

**The cache follows the beams.** After the sort, new beam `j` extends old row
`parents[j]`. Several new beams usually share a parent and some old rows have no
child. The token lists are rebuilt from `parents`, and the cache must be rebuilt the
same way, `cache.select(parents)`, before the new tokens are fed. The invariant:
*row `j` of the cache holds exactly the prompt plus the tokens of live beam `j`.*
The same call expands the prompt's single row to `n_beams` rows after step 1, which
is why the prompt is processed once and not once per beam.

**Finishing and stopping.** A candidate that ends with EOS leaves the beam and is
scored $S / L^{\alpha}$. Without the division ($\alpha = 0$) every extra token makes
a hypothesis less probable and short outputs win; $\alpha = 1$ compares average
log-probability per token. The normalized score is used only for finished
hypotheses. Live beams are always compared by $S$: within a step they all have the
same length, so normalizing would not change their order, and carrying a normalized
value into the next step would be adding log-probabilities to an average.

Two rules keep the search honest. Only a candidate ranked below `n_beams` may
finish: an EOS at rank `n_beams + 2` is a hypothesis that pruning would have thrown
away. And `2 * n_beams` candidates are taken so that `n_beams` live beams always
remain: each beam contributes at most one EOS candidate, so at most `n_beams` of the
`2 * n_beams` can be EOS.

The stop rule is a bound. Every descendant of a live beam has $S' \le S$ (each
token adds a log-probability $\le 0$) and length $L \le$ `max_new_tokens`. Since
$S' \le 0$ and $\alpha \ge 0$,

$$
\frac{S'}{L^{\alpha}} \;\le\; \frac{S'}{\texttt{max\_new\_tokens}^{\alpha}}
\;\le\; \frac{S_{\text{best}}}{\texttt{max\_new\_tokens}^{\alpha}} .
$$

If that bound does not exceed the worst score in a full finished set, nothing that
is still live can enter the set, and stopping does not change the result. The rule
is purely a saving: the cost judge is what checks it.

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

Step 1 is not a special case: one live beam, the empty one, with a one-row table.
The tensor work per step is one log-softmax and one sort; the Python loop touches
`2 * n_beams` candidates. `list.sort` is stable, which implements "of two equal
scores, the earlier one stays ahead" for free.

## Common mistakes

- **Not reordering the cache.** Expanding the cache once and then feeding the new
  tokens row by row is right only while new beam `j` happens to descend from old row
  `j`. After the first reshuffle each row continues another beam's history. Nothing
  fails; the hypotheses are simply not the most probable ones.
- **Letting finished hypotheses keep expanding.** If an EOS candidate stays in the
  live list it occupies a beam slot and the model is asked what follows EOS. Real
  code avoids this either by removing it (as here) or by forcing its score to
  $-\infty$ for every continuation.
- **Normalizing before pruning.** Storing $S / L^{\alpha}$ as the beam's running
  score means the next step adds a log-probability to a normalized number.
- **Taking only `n_beams` candidates.** Every EOS in the top `n_beams` then costs a
  live beam, and the search narrows toward greedy just when it has found its first
  results.
- **Finishing every EOS among the candidates.** The spare candidates exist to refill
  the beam, not to finish; an EOS ranked below `n_beams` was pruned.
- **Counting the prompt in the length.** $S$ covers generated tokens only, so the
  length must too. With a long prompt, $(P + L)^{\alpha}$ is nearly constant and the
  penalty silently stops working. (This is why the beam code of transformers
  subtracts a `decoder_prompt_len` wherever it uses a length.)
- **Raw logits as scores.** Sums of unnormalized logits are not comparable across
  beams.
- **Stopping when the finished set is full.** `n_beams` finished hypotheses are not
  necessarily the best: in the second example of the statement, the hypothesis that
  ends up second finishes two steps later.
- **Bounding with the current length.** $S_{\text{best}} / t^{\alpha}$ is not an
  upper bound when $\alpha > 0$: a longer hypothesis divides by more. It stops the
  same example at step 3.
- **Never stopping, or re-feeding prefixes.** Both return the right hypotheses and
  fail the cost limits: calls after the result is decided, or
  $O(\text{steps}^2)$ tokens.

## Complexity

With $s$ steps: $s$ model calls and $P + n_{\text{beams}} (s - 1)$ tokens, against
$P + s - 1$ for greedy decoding, so beam search costs about `n_beams` times the
decode work, in batches of `n_beams`. Each step sorts $n_{\text{beams}} \cdot V$
numbers; a top-k selection of $2 n_{\text{beams}}$ makes that linear. `select`
copies the cache rows, $O(n_{\text{beams}} \cdot \text{length})$ per layer and step,
which in real engines is the expensive part of beam search and the reason some
implementations store beams as a tree of shared cache blocks instead.

Note what the exact bound costs with $\alpha > 0$: with `n_beams = 1` and
$\alpha = 1$ the search does not stop at the first EOS, because a live beam with a
better average could still appear. Exact stopping and length normalization pull in
opposite directions.

## In the wild

This problem follows the beam search of Hugging Face transformers
(`generation/utils.py`, `_beam_search`): candidates
`beams_to_keep = 2 * num_beams`, only the top `num_beams` may finish, finished
scores `log_prob / length ** length_penalty` with the length counted without the
prompt, and the best `num_beams` finished hypotheses kept. The stop rule here is its
`early_stopping="never"` mode, which the source describes as the canonical
algorithm. Its default, `early_stopping=False`, uses the current length in the
bound: a heuristic that stops sooner and can miss a better hypothesis (the
"bounding with the current length" mistake above, shipped on purpose for backward
compatibility). `early_stopping=True` stops as soon as `num_beams` hypotheses have
finished. Its cache reordering is the same gather as `cache.select`, applied to
every layer's keys and values.

Among LLM servers, beam search is rare: vLLM implements it in its entrypoint
layer (`vllm/entrypoints/generate/beam_search`), on top of ordinary generation
requests instead of inside the engine, and llama.cpp's public API has no beam
search. Sampling dominates open-ended text
generation, where the most probable sequence tends to be bland or repetitive. Beam
search remains standard where there is one right answer: speech recognition
(the Whisper command line defaults to a beam size of 5), translation, and constrained
decoding.
