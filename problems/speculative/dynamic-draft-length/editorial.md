## The idea

Each round of speculative decoding pays for $k$ draft steps and one target call, and produces the accepted prefix plus one target token. Given a per-token acceptance rate $\alpha$, the optimal draft length $k$ grows quickly with $\alpha$ (`speculative-speedup-model`). Near $\alpha = 1$, a long draft saves many target calls. At $\alpha = 0.3$, even a single drafted token can cost more than it saves. Because $\alpha$ shifts within a single response as the text moves between predictable and unpredictable spans, any fixed $k$ is suboptimal much of the time.

The policy combines two cheap online signals to adapt $k$ as decoding runs.

**The outcome of the previous round.** If every drafted token was accepted, the draft was likely too short, so the policy grows $k$. If any token was rejected, every token drafted after the first mismatch was wasted, so the policy shrinks $k$. Hugging Face's rule is additive and asymmetric, adding 2 on a full acceptance and subtracting 1 on a rejection, with a floor of 1 so drafting never shuts off completely. This asymmetry biases $k$ upward. A run of $n$ full acceptances raises $k$ by $2n$, and it takes $2n$ rejections to bring it back down. That matches the cost structure when the draft model is much cheaper than the target, because a slightly over-long draft wastes cheap steps whereas a too-short draft wastes an expensive target call.

**The draft model's own confidence.** The round outcome arrives one round late and changes $k$ gradually. In contrast, the draft model's probability for each token it proposes is available immediately. When the draft model is uncertain, it is usually wrong. Stopping the draft at the first low-confidence token cuts off wasted steps right where the draft starts to fail, without waiting for verification.

These two signals interact through one key rule. The growth check compares `n_accepted` against `n_drafted`, the number of tokens **actually drafted**, rather than against `k`. If confidence stopping cuts a draft to three tokens and all three are accepted, the round is still a full acceptance and grows $k$. With confidence stopping handling per-token early exits, $k$ acts as an upper cap that drifts upward on easy spans and drops only when confident drafts are rejected.

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

The confidence computation is stable softmax (`stable-softmax`) evaluated only at the maximum entry. Subtracting `top` makes the numerator $e^0 = 1$ and keeps every exponent non-positive.

## Common mistakes

- **Growing on any acceptance.** Checking `n_accepted > 0` grows `k` even when a draft of eight tokens lost seven of them. Only a fully accepted draft signals that `k` was too short.
- **Never shrinking below the initial `k`.** Clamping at the initial `k` instead of `1` disables the protection on hard spans. With an initial `k` of 5, every round on an unpredictable span wastes four draft steps.
- **Comparing `threshold` against a raw logit or overflowing `math.exp`.** A logit like `3.0` is not a probability, so it must be converted via softmax. Subtract the maximum logit before calling `math.exp`, or large logits like $10^4$ will raise `OverflowError`.
- **Dropping the token that triggered the stop.** In this policy, the confidence check runs after appending the token, so that token is drafted, paid for, and verified. Discarding it before verification is a different policy (used by `llama.cpp`, described below) and changes all round counts.
- **Comparing `n_accepted` with `k`.** Checking `n_accepted == k` breaks once confidence stopping is enabled. A draft shortened by low confidence can never reach `k`, so `k` would only ever shrink during easy spans.
- **Drafting the final token.** Using `min(k, total - position)` drafts a token at the last position that can never be used, because the target's own token already finishes the sequence.
- **Counting matches after a mismatch.** Verification accepts only a contiguous prefix of matches. Any match after the first mismatch was conditioned on a wrong draft token.
- **Small rule deviations (`<=` instead of `<`, `+1` instead of `+2`, or no floor at `1`).** None of these change the decoded text, so they only show up in the schedule and cost metrics.
- **Forgetting the target token or mischarging target cost.** The target model runs once per round regardless of `n_drafted`, costs `c_target`, and always contributes one token to `position`.

## Complexity

`simulate` visits each drafted position once per round that drafts it. Because rejected positions can be drafted again in later rounds, the time complexity is $O(T \cdot \bar k)$ in the worst case (where $\bar k$ is the average draft length) and $O(T)$ when most drafts are accepted.

## In the wild

**Hugging Face `transformers` (assisted generation).** `GenerationConfig` documents `num_assistant_tokens_schedule="heuristic"` as increasing `num_assistant_tokens` by 2 when all speculative tokens are correct and reducing it by 1 otherwise. `AssistedCandidateGenerator.update_candidate_strategy` implements this by comparing the number of matches against the number of candidate tokens in the round and clamping at 1. Its `ConfidenceCriteria` stopping rule takes the softmax of the assistant's latest scores, reads the probability of the last generated token, and stops when `p < assistant_confidence_threshold`. This problem models both mechanisms. When `scikit-learn` is installed, `transformers` also dynamically re-tunes the confidence threshold during generation using an ROC-based heuristic inspired by "Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models."

**`llama.cpp`.** `llama.cpp` uses a different design without round-to-round length adaptation. It configures a maximum draft length, a minimum draft length, and a minimum draft probability (`--spec-draft-n-max`, `--spec-draft-n-min`, and `--spec-draft-p-min`, previously `--draft-max`, `--draft-min`, and `--draft-p-min`). In `common/speculative.cpp`, the draft loop checks the sampled token's probability against `p_min` *before* appending it, so the low-confidence token is discarded, and any draft shorter than the minimum length is dropped entirely.

**What the numbers say.** On the phased traces in this problem, the additive rule alone without confidence stopping beats the best constant `k` by only a few percent when `c_draft` is one-tenth of `c_target`, and loses to it when drafting is twice as expensive. Adding confidence stopping lowers the cost to roughly $0.8$ of the best constant `k`. A large constant `k` paired with the same confidence stop performs slightly better still on these synthetic traces. Confidence stopping does most of the work when confidence is well calibrated, while the additive rule protects against poorly calibrated draft models.
