## The idea

All five stages fit into one decoding loop by separating three concerns: who proposes draft tokens, how those proposals are verified, and how each model's KV cache tracks the sequence.

![One round. A proposer drafts up to $k$ tokens, the target scores them in one call, and the verifier accepts a prefix.](assets/engine.svg)

**Cache management (stage 1).** Instead of tracking two caches and two lengths manually as in `speculative-decode-loop`, wrap each model together with its cache and the number of sequence tokens currently cached. Give the wrapper two methods. `feed(tokens)` passes only the unseen suffix `tokens[self.length:]` to the model, and `rollback(n)` truncates the cache to `n` tokens when `self.length > n`. With this abstraction, every cache case works automatically. The first draft call of a round catches up on whatever verified tokens the draft missed (two after a full acceptance, or many after lookup rounds). The target call feeds one verified token plus the drafts. Calling `rollback(len(seq) - 1)` on both models at the end of a round removes rejected drafts and leaves the newest verified token outside the cache so the next round can score what follows it.

**Verification (stage 2).** Greedy verification compares argmaxes. Sampling verification follows `speculative-sampling-block`. It accepts draft $j$ with probability $\min(1, p_j[x_j] / q_j[x_j])$, stops at the first rejection, and samples one more token from $\max(0, p_n - q_n)$ on rejection or from $p_m$ on full acceptance. For any token $v$, the probability of proposing and accepting it is $\min(p[v], q[v])$, and the probability of rejecting and drawing $v$ from the residual is $(1 - \sum_i \min(p_i, q_i)) \cdot \max(0, p[v] - q[v]) / \sum_i \max(0, p_i - q_i)$. Because the two normalizers cancel, the total probability is exactly $p[v]$. Indexing uniforms by round ensures every draw is independent, even when a rejected position is drafted again in the next round.

**Adaptive draft length (stage 3).** Updating $k$ takes two lines after verification. Compare $n$ against the number of tokens actually drafted ($m$), not against $k$. Near the end of generation or after a short lookup match, $m$ can be smaller than $k$, and accepting all $m$ tokens should still count as a full acceptance.

**Prompt lookup (stage 4).** Lookup returns tokens without probability distributions. A deterministic copy is equivalent to a one-hot distribution $q_j$ that puts all its mass on the copied token $x_j$. Passing one-hot rows as $q$ keeps the verifier completely unchanged. Greedy verification ignores $q$, while sampling verification accepts with probability $p_j[x_j]$ and sets the residual to $p_n$ with entry $x_n$ zeroed out. The draft model is skipped during lookup rounds, and its wrapper catches up automatically on the next draft round.

**Batched verification (stage 5).** After a verification step, different rows accept different numbers of drafts, while the shared KV cache has a single length across the batch. Two properties solve this. Because the model is causal, right-padding shorter rows does not affect the logits of their real tokens. Because any valid shared prefix can be kept, each round feeds every active row's verified tokens beyond the common cached prefix $c$ followed by its drafts, reads each row's verification logits at its own offset, and truncates the shared cache to the shortest valid length among the remaining rows. Faster rows reprocess their lead tokens, which costs extra tokens rather than extra rounds. Truncating every row to the batch minimum acceptance instead wastes rounds (31 vs. 21 in Example 5) and degrades further as batch size grows.

## Reference solution (PyTorch)

```python
class _Cached:
    def __init__(self, lm):
        self.lm, self.cache, self.length = lm, None, 0

    def feed(self, tokens):
        logits, self.cache = self.lm(tokens[self.length :], self.cache)
        self.length = len(tokens)
        return logits

    def rollback(self, n):
        if self.length > n:
            self.cache, self.length = self.cache.truncate(n), n


def generate(self, prompt, max_new_tokens, uniforms=None):
    seq = list(prompt)
    target, draft = _Cached(self.target), _Cached(self.draft)
    k = self.k
    self.rounds = []
    while len(seq) - len(prompt) < max_new_tokens:
        limit = min(k, max_new_tokens - (len(seq) - len(prompt)) - 1)
        u = None if uniforms is None else uniforms[len(self.rounds)].tolist()
        proposer, drafts, q = self._propose(draft, seq, limit, u)
        logits = target.feed(seq + drafts)[-(len(drafts) + 1) :]
        if u is None:
            n, token = self._verify_greedy(logits, drafts)
        else:
            n, token = self._verify_sampled(ops.softmax(logits).tolist(), q, drafts, u)
        seq += [*drafts[:n], token]
        self.rounds.append((proposer, len(drafts), n))
        if self.adaptive and drafts:
            k = k + 2 if n == len(drafts) else max(1, k - 1)
        target.rollback(len(seq) - 1)
        draft.rollback(len(seq) - 1)
    return seq[len(prompt) :]


def _propose(self, draft, seq, limit, u):
    if limit and self.lookup is not None:
        found = _lookup(seq, *self.lookup, limit)
        if found:
            vocab = range(self.target.vocab_size)
            return "lookup", found, [[float(v == x) for v in vocab] for x in found]
    drafts, rows = [], []
    while len(drafts) < limit:
        logits = draft.feed(seq + drafts)[-1]
        if u is None:
            drafts.append(int(torch.argmax(logits)))
        else:
            rows.append(ops.softmax(logits).tolist())
            drafts.append(_inverse_cdf(rows[-1], u[len(drafts)][0]))
    return "draft", drafts, rows
```

`_verify_sampled` implements the block rule from the last hint, `_lookup` follows `prompt-lookup-draft`, and `generate_batch` uses one `_Cached` draft wrapper per row with the batched target step from the last hint. Because token decisions are scalar logic, the reference converts each call's probabilities to Python floats once with `tolist()`. This avoids repeated host-device round trips per token.

## Common mistakes

- **Stale target cache.** Leaving rejected drafts in the target cache produces fluent text that no longer matches the target model.
- **Stale draft cache.** Output tokens remain correct, but the draft model continues from rejected tokens, causing acceptance to collapse and changing the `rounds` trace.
- **Refeeding the whole sequence.** Recomputing from token 0 produces the right tokens but violates the token-count limits with quadratic work.
- **Unnormalized residual.** Sampling $\max(0, p - q)$ as if it summed to 1 runs off the end of the cumulative sums whenever the uniform exceeds the residual total, dumping the missing probability mass onto the clamped fallback token.
- **Greedy drafts in sampling mode.** The acceptance ratio assumes $x \sim q$. Testing an argmax draft with $q[x]$ biases the output toward the draft model's top tokens.
- **Counting total passes instead of a prefix.** A draft that passes after an earlier draft failed was conditioned on a rejected token and cannot be accepted.
- **Broken $k$ updates.** Leaving $k$ fixed or allowing $k$ to drop to 0 without the floor of 1 disables speculation for the rest of the sequence.
- **Trusting copied tokens.** Emitting prompt-lookup copies without target verification corrupts the output whenever the text stops repeating.
- **Allowing a rejected copy to be redrawn.** Sampling the replacement from $p$ instead of $p$ with $x_n$ zeroed out gives $x_n$ probability $p[x_n] + (1 - p[x_n])\,p[x_n]$.
- **Picking the latest lookup match.** Choosing the most recent n-gram match instead of the earliest changes the drafts and the `rounds` trace.
- **Incorrect batch handling.** Truncating all rows to the minimum acceptance wastes rounds, decoding rows sequentially exceeds the call limit, and reading verification logits from the right edge of the padded tensor breaks shorter rows.

## Complexity

Across $R$ rounds with $D$ total drafted tokens and $D_m$ model-drafted tokens, the target processes $P + D + R$ tokens in $R$ calls and the draft processes at most $P + N + D_m$ tokens. Standard autoregressive decoding requires $N$ target calls. Direct prompt lookup takes $O(L \cdot n_g)$ list comparisons per round on a sequence of length $L$, though a suffix automaton can reduce this to constant time per token. In batched mode, the number of target calls equals the round count of the slowest row. Total target tokens increase with the length gap between rows because faster rows reprocess their lead each round, which production engines avoid using per-row cache lengths and attention masks.

## In the wild

- The speculative rejection-sampling rule and its exactness proof come from "Fast Inference from Transformers via Speculative Decoding" (ICML 2023, arXiv:2211.17192) and "Accelerating Large Language Model Decoding with Speculative Sampling" (arXiv:2302.01318).
- The adaptive schedule (+2 on full acceptance, -1 otherwise) matches Hugging Face `transformers` assisted generation (`num_assistant_tokens_schedule="heuristic"`). `problems/speculative/dynamic-draft-length` pairs it with a confidence stop, which this capstone omits because it did not lower cost on the kit's model pairs at any tested threshold.
- Prompt lookup decoding (`github.com/apoorvumang/prompt-lookup-decoding`) appears as `prompt_lookup_num_tokens` in `transformers` and as the n-gram speculator in vLLM.
- Batched speculative decoding with ragged per-row acceptance is explored in "BASS: Batched Attention-optimized Speculative Sampling" (arXiv:2404.15778), which addresses the padding and ragged-tensor overhead of rectangular caches.
