# BUG: runs the draft on the whole sequence for every proposal instead of continuing from its cache: the right tokens at a cost that grows quadratically.
import torch

from whetstone.kit import TinyLM


def speculative_greedy(
    target: TinyLM,
    draft: TinyLM,
    prompt: list[int],
    k: int,
    max_new_tokens: int,
    eos_id: int | None,
) -> list[int]:
    seq = list(prompt)
    t_cache = None
    t_len = 0
    while True:
        n_draft = min(k, max_new_tokens - (len(seq) - len(prompt)) - 1)
        drafts: list[int] = []
        for _ in range(n_draft):
            logits, _ = draft(seq + drafts)
            drafts.append(int(torch.argmax(logits[-1])))
        logits, t_cache = target((seq + drafts)[t_len:], t_cache)
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
        t_len = len(seq) - 1
        t_cache = t_cache.truncate(t_len)
