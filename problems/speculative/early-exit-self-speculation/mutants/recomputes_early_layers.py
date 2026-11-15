# BUG: treats the early exit as a separate draft model: verification runs all layers over the drafted positions through a cache of its own, so the early layers are evaluated twice on every position.
import torch

from whetstone.kit import LayeredLM


def self_speculative_greedy(
    lm: LayeredLM, prompt: list[int], exit_layer: int, k: int, max_new_tokens: int
) -> list[int]:
    seq = list(prompt)
    draft_cache = lm.empty_cache()  # layers 0 .. E - 1, used by the draft loop
    full_cache = lm.empty_cache()  # all layers, used by verification
    while True:
        seen = full_cache.length
        n_draft = min(k, max_new_tokens - (len(seq) - len(prompt)) - 1)

        feed, position = seq[seen:], seen
        drafts: list[int] = []
        for _ in range(n_draft):
            h, draft_cache = lm.layers(lm.embed(feed, offset=position), 0, exit_layer, draft_cache)
            position += len(feed)
            feed = [int(torch.argmax(lm.head(lm.norm(h[-1:]))))]
            drafts.append(feed[0])

        # Verify: the whole model, from the embeddings, on the new positions.
        h = lm.embed(seq[seen:] + drafts, offset=seen)
        h, full_cache = lm.layers(h, 0, lm.n_layers, full_cache)
        greedy = torch.argmax(lm.head(lm.norm(h[-(n_draft + 1) :])), dim=-1).tolist()
        n = 0
        while n < n_draft and drafts[n] == greedy[n]:
            n += 1
        seq += [*drafts[:n], greedy[n]]
        if len(seq) - len(prompt) == max_new_tokens:
            return seq[len(prompt) :]

        full_cache = full_cache.truncate(len(seq) - 1)
        draft_cache = draft_cache.truncate(len(seq) - 1)
