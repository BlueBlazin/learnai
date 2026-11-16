# BUG: runs the target on the whole sequence in every round instead of continuing from its cache: the right tokens at a cost that grows quadratically.
import torch

from learnai.kit import TinyLM


def speculative_greedy(
    target: TinyLM,
    draft: TinyLM,
    prompt: list[int],
    k: int,
    max_new_tokens: int,
    eos_id: int | None,
) -> list[int]:
    seq = list(prompt)
    d_cache = None
    d_len = 0
    while True:
        n_draft = min(k, max_new_tokens - (len(seq) - len(prompt)) - 1)
        drafts: list[int] = []
        for _ in range(n_draft):
            logits, d_cache = draft((seq + drafts)[d_len:], d_cache)
            d_len = len(seq) + len(drafts)
            drafts.append(int(torch.argmax(logits[-1])))
        logits, _ = target(seq + drafts)
        greedy = torch.argmax(logits[-(n_draft + 1) :], dim=-1).tolist()
        n = 0
        while n < n_draft and drafts[n] == greedy[n]:
            n += 1
        new = [*drafts[:n], greedy[n]]
        if eos_id in new:
            new = new[: new.index(eos_id) + 1]
        seq += new
        if new[-1] == eos_id or len(seq) - len(prompt) == max_new_tokens:
            return seq[len(prompt) :]
        d_len = min(d_len, len(seq) - 1)
        d_cache = d_cache.truncate(d_len)
