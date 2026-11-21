## The idea

Five problems, one loop. What makes them compose is a separation that none of the
component problems needed on its own: **who proposes** is one decision, **how a
proposal is verified** is another, and **where each cache stands** is bookkeeping
that neither of them should see.

![One round. A proposer drafts up to $k$ tokens, the target scores them in one call, and the verifier accepts a prefix.](assets/engine.svg)

**The caches (stage 1).** `speculative-decode-loop` tracks two caches and two
lengths by hand. Put a model, its cache, and "how many tokens of the sequence the
cache covers" into one object with two methods. `feed(tokens)` passes the model
whatever of `tokens` it has not seen; `rollback(n)` truncates to `n` if the cache is
longer. Then every subtlety of that problem is a consequence, not a case: the first
draft call of a round feeds the verified tokens the draft missed (two after a fully
accepted round, many after lookup rounds); the target call feeds one verified token
plus the drafts; and `rollback(len(seq) - 1)` on both removes rejected drafts and
keeps the newest token outside, because the next round starts from its logits.
Truncating a cache costs nothing; refeeding it costs everything the speculation
saved.

**The verifier (stage 2).** Greedy verification compares argmaxes. Sampling
verification is `speculative-sampling-block`: accept draft $j$ with probability
$\min(1, p_j[x_j] / q_j[x_j])$, stop at the first rejection, and emit one more token
from the residual $\max(0, p_n - q_n)$ or, after a full acceptance, from $p_m$.
The token at each position is then distributed as $p$: for a token $v$, the
probability of proposing and accepting it is $\min(p[v], q[v])$, and the probability
of rejecting something and then drawing $v$ is
$(1 - \sum_i \min(p_i, q_i)) \cdot \max(0, p[v] - q[v]) / \sum_i \max(0, p_i - q_i)$;
the two normalizers are equal, so the sum is $p[v]$. Exactness needs every uniform
to be fresh, which is why they are indexed by round: a token position can be drafted
in one round, thrown away, and drafted again in the next, and reusing its uniform
would correlate the second draft with the decisions that led to it.

**The draft length (stage 3)** is two lines after the verification. It compares
the number accepted with the number *drafted*: near the end of the output, or after
a short lookup copy, fewer than $k$ tokens are drafted, and a full acceptance of
those should still count as one.

**The second proposer (stage 4).** Lookup returns tokens without distributions. The
insight that makes it fit is that a deterministic proposal *is* a distribution: all
mass on one token. Hand the verifier one-hot rows as $q$ and nothing else changes.
Greedy verification never looked at $q$; sampling verification accepts with
probability $p[x]$ and its residual is $p$ with the entry of $x$ removed, both
straight from the general formulas. The draft model is not called in a lookup
round, and its wrapper needs no special case: its cache is simply further behind
the next time it is fed.

**The batch (stage 5).** After one verification, row 0 has accepted nothing and row
1 everything. The kit's cache, like most tensors, has one length. Two facts resolve
it. The model is causal, so a row can be padded on the right with anything: the
logits of its real positions do not change. And a cache may be cut to any prefix
that is valid for every row. So each round feeds, per row, the verified tokens
beyond the common prefix plus the drafts, reads each row's logits at its own offset,
and cuts the cache to the shortest valid row. Rows that got ahead feed their lead
again; that costs tokens, not rounds. The alternative that keeps everything
rectangular, discarding what the luckier rows accepted, costs rounds: 31 instead of
21 in Example 5, and worse as the batch grows.

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

`_verify_sampled` is the block rule (see the last hint), `_lookup` is
`prompt-lookup-draft`, and `generate_batch` is the round of the last hint with one
`_Cached` draft per row. The decisions about single tokens are scalar logic, so the
reference brings each call's probabilities to Python once (`tolist`) and compares
plain floats: a few tensor operations per model call, which matters when a GPU
framework pays a round trip for every value it hands back.

## Common mistakes

- **A stale target cache.** Keeping rejected drafts in the target's cache gives
  fluent output that is not the target's.
- **A stale draft cache.** The output stays right, the draft model continues from
  tokens that were rejected, acceptance collapses, and only the round count shows it.
- **Refeeding the whole sequence** to either model: correct tokens, quadratic work.
- **The residual not normalized.** Sampling $\max(0, p - q)$ as if it summed to 1
  runs off the end of the cumulative sums whenever the uniform exceeds the total, and
  whatever catches that (a clamp, usually) receives the missing mass.
- **Greedy drafts in sampling mode.** The acceptance test assumes $x \sim q$; an
  argmax draft is tested as if it had been sampled and the output leans toward the
  draft model's favorite tokens.
- **Counting passes instead of a prefix.** A draft that passes its test after an
  earlier one failed was conditioned on a rejected token.
- **A fixed $k$**, or a $k$ that shrinks to zero without the floor of 1 and turns
  speculation off for the rest of the output.
- **Copied tokens trusted.** "It occurred before" is a guess; emitting it unverified
  changes the output wherever the text does not repeat.
- **A rejected copy that can come back.** Drawing the replacement from $p$ instead
  of from $p$ without the rejected token gives that token probability
  $p[x] + (1 - p[x])\,p[x]$.
- **The latest occurrence** instead of the earliest in the lookup: other drafts,
  other rounds.
- **Rows cut to the batch minimum**, rows decoded one at a time, or a row's
  verification logits read from the end of the padded call, where only the longest
  row has them.

## Complexity

With $R$ rounds and $D$ drafted tokens, the target processes $P + D + R$ tokens in
$R$ calls and the draft at most $P + N + D_m$; plain decoding needs $N$ target
calls. Lookup is $O(L \cdot n)$ list work per round for a sequence of length $L$
(the direct search; a suffix automaton makes it constant per token). In a batch the
number of calls is that of the slowest row, and the token count grows with how far
the rows drift apart, because the lead of the faster rows is fed again each round;
an engine with per-row cache lengths (an attention mask per row) avoids even that.

## In the wild

- The acceptance rule and its proof are in "Fast Inference from Transformers via
  Speculative Decoding" (ICML 2023, arXiv:2211.17192) and "Accelerating Large
  Language Model Decoding with Speculative Sampling" (arXiv:2302.01318).
- The heuristic length schedule is that of `transformers`' assisted generation
  (`num_assistant_tokens_schedule="heuristic"`: plus 2 after a fully accepted round,
  minus 1 otherwise); `problems/speculative/dynamic-draft-length` covers it together
  with the confidence stop, which this capstone leaves out: on the kit's model pairs
  it did not lower the cost at any threshold that was tried.
- Prompt lookup decoding (`github.com/apoorvumang/prompt-lookup-decoding`) is
  `prompt_lookup_num_tokens` in `transformers` and the n-gram speculator of vLLM.
- Batched speculative decoding with per-row acceptance is the subject of "BASS:
  Batched Attention-optimized Speculative Sampling" (arXiv:2404.15778), which
  discusses the padding and ragged-tensor handling that the rectangular cache forces
  here.
