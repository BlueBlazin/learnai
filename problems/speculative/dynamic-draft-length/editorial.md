## The idea

One round of speculative decoding costs $k$ draft steps and one target call, and
yields the accepted prefix plus one token. With a per-token acceptance rate
$\alpha$, the best $k$ grows quickly with $\alpha$ (`speculative-speedup-model`):
near $\alpha = 1$ the longest possible draft is best, and at $\alpha = 0.3$ even one
drafted token can cost more than it saves. The acceptance rate is not a property of
a model pair. It changes within one response, from a span the draft can copy to a
span it cannot predict, so a constant $k$ is wrong most of the time in one direction
or the other.

Two cheap signals are available online, and the policy uses both.

**The outcome of the last round.** If every drafted token was accepted, the draft
was too short: grow. If a token was rejected, everything drafted after it was
wasted: shrink. Hugging Face's rule is additive and asymmetric, $+2$ on success and
$-1$ on failure, with a floor of 1 so that drafting never stops. The asymmetry
biases $k$ upward: a run of $n$ full acceptances raises $k$ by $2n$, and it takes
$2n$ rejections to undo it. That fits the cost structure when the draft model is
much cheaper than the target, since an over-long draft wastes cheap steps while a
too-short draft wastes an expensive target call.

**The draft's own confidence.** The outcome signal arrives one round late and moves
$k$ by a step at a time. The draft model's probability of its own token is available
immediately, for every token, and it is informative: a draft that is unsure is
usually wrong. Ending the draft at the first low-confidence token cuts a long draft
exactly where it starts to go wrong, without waiting for the next round.

The two interact through one detail: the growth rule compares the accepted count
with the number of tokens **actually drafted**. A draft that the confidence rule cut
to three tokens, all accepted, is a success and grows $k$, even though the round
never drafted $k$ tokens. With confidence stopping doing the fine control, $k$
becomes a ceiling that drifts upward during easy spans and comes down only when
confident drafts get rejected.

## Reference solution (Python)

```python
def should_stop_drafting(logits, threshold):
    top = max(logits)
    confidence = 1.0 / sum(math.exp(x - top) for x in logits)
    return confidence < threshold


def next_k(k, n_accepted, n_drafted):
    if n_accepted == n_drafted:
        return k + 2
    return max(1, k - 1)


def simulate(trace, k, schedule, threshold, c_draft, c_target):
    total, position, cost, draft_tokens, ks = len(trace), 0, 0, 0, []
    while position < total:
        ks.append(k)
        limit = min(k, total - position - 1)
        n_drafted = 0
        while n_drafted < limit:
            _, logits = trace[position + n_drafted]
            n_drafted += 1
            if should_stop_drafting(logits, threshold):
                break
        n_accepted = 0
        while n_accepted < n_drafted and trace[position + n_accepted][0]:
            n_accepted += 1
        cost += n_drafted * c_draft + c_target
        draft_tokens += n_drafted
        position += n_accepted + 1
        if schedule == "heuristic" and n_drafted > 0:
            k = next_k(k, n_accepted, n_drafted)
    return {"cost": cost, "rounds": len(ks), "draft_tokens": draft_tokens, "ks": ks}
```

The confidence is the stable softmax of `stable-softmax` reduced to one entry:
after subtracting the maximum, the numerator is $e^0 = 1$.

## Common mistakes

- **Growing on any acceptance.** `n_accepted > 0` is true for a draft of eight
  tokens that lost seven. The signal for "the draft was too short" is that nothing
  was rejected.
- **Never shrinking below the initial `k`.** Treating the configured value as a
  floor looks conservative and removes the half of the policy that protects hard
  spans: with an initial $k$ of 5, every round in a span the draft cannot predict
  wastes four draft steps.
- **Comparing the confidence with a logit.** A logit is not a probability; `3.0`
  is "above 0.4" for every distribution. The comparison needs the softmax, and the
  softmax needs the shift by the maximum, or `math.exp` overflows on large logits.
- **Dropping the token that triggered the stop.** In this policy the criterion is
  evaluated after the token has been appended, so the token is drafted, paid for,
  and verified. Stopping *before* it is a different policy (llama.cpp's, below) with
  different counts.
- **Comparing with `k`.** `n_accepted == k` was natural before confidence stopping
  existed. With it, a shortened draft can never satisfy the test, so $k$ only ever
  shrinks during the spans where the draft is doing well.
- **Drafting the last token.** With `min(k, remaining)` the final drafted token can
  never be used: the target's own token completes the output first.
- **Counting matches after a mismatch.** Verification keeps a prefix. A match after
  the first mismatch was computed from a wrong context.
- **`<=` for `<`**, **a step of $+1$**, **no floor at 1**: each changes the schedule,
  none changes the text, and the cost check is what notices.
- **Forgetting the target's token**, or charging the target per drafted token: the
  target runs once per round and always yields one token of its own.

## Complexity

`simulate` reads each position at most once per round that drafts it. A position can
be drafted again after a rejection, so the work is $O(T \cdot \bar k)$ in the worst
case and $O(T)$ when most drafts are accepted.

## In the wild

**Hugging Face `transformers`** (assisted generation). `GenerationConfig` documents
`num_assistant_tokens_schedule="heuristic"` as "When all speculative tokens are
correct, increase `num_assistant_tokens` by 2 else reduce by 1", and
`AssistedCandidateGenerator.update_candidate_strategy` implements it as
`num_assistant_tokens += 2` when the number of matches equals the number of
candidate tokens of the round, else `max(1, num_assistant_tokens - 1)`. The
confidence rule is the stopping criterion `ConfidenceCriteria`: it takes the softmax
of the assistant's last scores, reads the probability of the last generated token,
and stops when `p < assistant_confidence_threshold`. This problem follows both. The
library additionally re-tunes the threshold during generation from the observed
matches (an ROC-based choice, when scikit-learn is installed), which it describes as
an unsupervised version of the method of "Dynamic Speculation Lookahead Accelerates
Speculative Decoding of Large Language Models"; that part is not modeled here.

**llama.cpp** takes the other design: no adaptation of the length between rounds,
but a maximum draft length, a minimum, and a minimum draft probability
(`--spec-draft-n-max`, `--spec-draft-n-min`, `--spec-draft-p-min`; older releases
spelled them `--draft-max`, `--draft-min`, `--draft-p-min`). Its draft loop in
`common/speculative.cpp` checks the probability of the sampled token against `p_min`
*before* adding the token, so the low-confidence token is discarded, and a draft
that ends up shorter than the minimum is thrown away as a whole.

**What the numbers say.** On the phased traces of this problem, the additive rule
alone, without the confidence stop, beats the best constant $k$ by only a few
percent when a draft step costs a tenth of a target call, and loses to it when the
draft is twice as expensive. Adding the confidence stop brings the cost to about
$0.8$ of the best constant $k$. A *constant* large $k$ with the same confidence stop
does slightly better still on these traces. The confidence signal carries most of
the benefit; the additive rule matters when confidence is poorly calibrated, which
synthetic traces with a clean confidence gap do not capture.
