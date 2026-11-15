# BUG: the early and the late layers keep separate caches and only the early one is rolled back after a rejection; the late layers keep the keys and values of rejected drafts and verify the next round against them.
import torch

from whetstone.kit import LayeredLM


def self_speculative_greedy(
    lm: LayeredLM, prompt: list[int], exit_layer: int, k: int, max_new_tokens: int
) -> list[int]:
    seq = list(prompt)
    early = lm.empty_cache()  # layers 0 .. E - 1
    late = lm.empty_cache()  # layers E .. L - 1
    while True:
        seen = early.length
        n_draft = min(k, max_new_tokens - (len(seq) - len(prompt)) - 1)

        feed, position = seq[seen:], seen
        exits, drafts = [], []
        while True:
            h, early = lm.layers(lm.embed(feed, offset=position), 0, exit_layer, early)
            exits.append(h)
            position += len(feed)
            if len(drafts) == n_draft:
                break
            feed = [int(torch.argmax(lm.head(lm.norm(h[-1:]))))]
            drafts.append(feed[0])

        h, late = lm.layers(torch.cat(exits), exit_layer, lm.n_layers, late)
        greedy = torch.argmax(lm.head(lm.norm(h[-(n_draft + 1) :])), dim=-1).tolist()
        n = 0
        while n < n_draft and drafts[n] == greedy[n]:
            n += 1
        seq += [*drafts[:n], greedy[n]]
        if len(seq) - len(prompt) == max_new_tokens:
            return seq[len(prompt) :]

        early = early.truncate(len(seq) - 1)
