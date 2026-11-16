from itertools import accumulate

import torch

from whetstone.kit import TinyLM, ops

# Decisions about single tokens are scalar logic. Each model call's probabilities are
# brought to Python once (`tolist`), and everything after that is plain arithmetic.


def _inverse_cdf(weights: list[float], u: float) -> int:
    """The first index whose cumulative weight exceeds u times the total."""
    cdf = list(accumulate(weights))
    return sum(c <= u * cdf[-1] for c in cdf)


def _lookup(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]:
    """`prompt-lookup-draft`: the tokens after the earliest earlier occurrence of the
    longest suffix that has one."""
    length = len(context)
    for n in range(min(max_ngram, length - 1), min_ngram - 1, -1):
        suffix = context[length - n :]
        for start in range(length - n):
            if context[start : start + n] == suffix:
                return context[start + n : start + n + k]
    return []


class _Cached:
    """A model, its KV cache, and how many tokens of the sequence the cache covers."""

    def __init__(self, lm: TinyLM):
        self.lm, self.cache, self.length = lm, None, 0

    def feed(self, tokens: list[int]) -> torch.Tensor:
        """Feed whatever of `tokens` the cache does not cover; one row of logits each."""
        logits, self.cache = self.lm(tokens[self.length :], self.cache)
        self.length = len(tokens)
        return logits

    def rollback(self, n: int) -> None:
        """Keep at most the first n tokens: rejected drafts must not stay in the cache."""
        if self.length > n:
            self.cache, self.length = self.cache.truncate(n), n


class SpeculativeEngine:
    def __init__(
        self,
        target: TinyLM,
        draft: TinyLM,
        *,
        k: int = 4,
        adaptive: bool = False,
        lookup: tuple[int, int] | None = None,
    ):
        self.target, self.draft = target, draft
        self.k = k
        self.adaptive = adaptive
        self.lookup = lookup
        self.rounds: list[tuple[str, int, int]] = []

    # -- one sequence (stages 1 to 4) -----------------------------------------------------

    def generate(
        self, prompt: list[int], max_new_tokens: int, uniforms: torch.Tensor | None = None
    ) -> list[int]:
        seq = list(prompt)  # the verified sequence
        target, draft = _Cached(self.target), _Cached(self.draft)
        k = self.k
        self.rounds = []
        while len(seq) - len(prompt) < max_new_tokens:
            # A round emits up to limit + 1 tokens, so never draft past the budget.
            limit = min(k, max_new_tokens - (len(seq) - len(prompt)) - 1)
            u = None if uniforms is None else uniforms[len(self.rounds)].tolist()
            proposer, drafts, q = self._propose(draft, seq, limit, u)
            # One target call: the verified tokens it has not seen, then every draft. Its
            # last len(drafts) + 1 rows score the draft positions and the one after them.
            logits = target.feed(seq + drafts)[-(len(drafts) + 1) :]
            if u is None:
                n, token = self._verify_greedy(logits, drafts)
            else:
                n, token = self._verify_sampled(ops.softmax(logits).tolist(), q, drafts, u)
            seq += [*drafts[:n], token]
            self.rounds.append((proposer, len(drafts), n))
            if self.adaptive and drafts:
                k = k + 2 if n == len(drafts) else max(1, k - 1)
            # Both caches may hold rejected drafts. The newest token stays outside: it
            # is fed next round, because its logits are what that round starts from.
            target.rollback(len(seq) - 1)
            draft.rollback(len(seq) - 1)
        return seq[len(prompt) :]

    def _propose(self, draft: _Cached, seq: list[int], limit: int, u: list | None):
        """(proposer, drafts, q): q has one row per draft, the distribution it came from."""
        if limit and self.lookup is not None:
            found = _lookup(seq, *self.lookup, limit)
            if found:
                # A copied token is a proposal made with probability 1: q is one-hot.
                vocab = range(self.target.vocab_size)
                return "lookup", found, [[float(v == x) for v in vocab] for x in found]
        drafts: list[int] = []
        rows: list[list[float]] = []
        while len(drafts) < limit:
            # Each call feeds what the draft's cache lacks: verified tokens it has not
            # seen, then its own previous proposal.
            logits = draft.feed(seq + drafts)[-1]
            if u is None:
                drafts.append(int(torch.argmax(logits)))
            else:
                rows.append(ops.softmax(logits).tolist())
                drafts.append(_inverse_cdf(rows[-1], u[len(drafts)][0]))
        return "draft", drafts, rows

    @staticmethod
    def _verify_greedy(logits: torch.Tensor, drafts: list[int]) -> tuple[int, int]:
        greedy = torch.argmax(logits, dim=-1).tolist()
        n = 0
        while n < len(drafts) and drafts[n] == greedy[n]:
            n += 1
        return n, greedy[n]  # the correction at the first mismatch, or the bonus token

    @staticmethod
    def _verify_sampled(p: list, q: list, drafts: list[int], u: list) -> tuple[int, int]:
        """`speculative-sampling-block` with the round's uniforms."""
        n = 0
        # Draft n passes when u < p / q, written without the division. The loop stops
        # at the first failure: later drafts were conditioned on a rejected token.
        while n < len(drafts) and u[n][1] * q[n][drafts[n]] < p[n][drafts[n]]:
            n += 1
        if n < len(drafts):
            # The residual of the rejected position; the sampler normalizes it.
            weights = [max(0.0, a - b) for a, b in zip(p[n], q[n], strict=True)]
        else:
            weights = p[n]  # everything accepted: an ordinary sample from the target
        return n, _inverse_cdf(weights, u[n][2])

    # -- a batch of sequences (stage 5) ---------------------------------------------------

    def generate_batch(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        prompt_len = len(prompts[0])
        seqs = [list(prompt) for prompt in prompts]
        drafters = [_Cached(self.draft) for _ in prompts]
        active = list(range(len(prompts)))  # rows still generating, in batch order
        cache, cached = None, 0  # the target's cache covers `cached` tokens of every active row
        while active:
            # Each row drafts for itself; only the target call is shared.
            drafts = []
            for b in active:
                limit = min(self.k, max_new_tokens - (len(seqs[b]) - prompt_len) - 1)
                row: list[int] = []
                while len(row) < limit:
                    row.append(int(torch.argmax(drafters[b].feed(seqs[b] + row)[-1])))
                drafts.append(row)
            # Rows differ in length. The model is causal, so padding on the right cannot
            # change the logits of the positions before it.
            rows = [seqs[b][cached:] + row for b, row in zip(active, drafts, strict=True)]
            width = max(len(row) for row in rows)
            padded = torch.tensor([row + [0] * (width - len(row)) for row in rows])
            logits, cache = self.target(padded, cache)
            valid = []  # per row: how many of its tokens have good entries in the cache
            for i, b in enumerate(active):
                first = len(seqs[b]) - cached - 1  # the row that scores the first draft
                n, token = self._verify_greedy(
                    logits[i, first : first + len(drafts[i]) + 1], drafts[i]
                )
                valid.append(len(seqs[b]) + n)
                seqs[b] += [*drafts[i][:n], token]
                drafters[b].rollback(len(seqs[b]) - 1)
            keep = [i for i, b in enumerate(active) if len(seqs[b]) - prompt_len < max_new_tokens]
            if not keep:
                break
            # One length for every row: the shortest valid prefix. A row that accepted
            # more feeds the difference again next round.
            cached = min(valid[i] for i in keep)
            if len(keep) < len(active):
                cache = cache.select(keep)
            cache = cache.truncate(cached)
            active = [active[i] for i in keep]
        return [seq[prompt_len:] for seq in seqs]
