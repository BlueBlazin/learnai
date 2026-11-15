# BUG: no cache is kept between rounds: every round feeds the whole verified sequence through all layers again, so the layer-token count grows quadratically.
import torch

from whetstone.kit import LayeredLM


def self_speculative_greedy(
    lm: LayeredLM, prompt: list[int], exit_layer: int, k: int, max_new_tokens: int
) -> list[int]:
    seq = list(prompt)
    while True:
        cache = lm.empty_cache()  # started again every round
        n_draft = min(k, max_new_tokens - (len(seq) - len(prompt)) - 1)

        feed, position = list(seq), 0
        exits, drafts = [], []
        while True:
            h, cache = lm.layers(lm.embed(feed, offset=position), 0, exit_layer, cache)
            exits.append(h)
            position += len(feed)
            if len(drafts) == n_draft:
                break
            feed = [int(torch.argmax(lm.head(lm.norm(h[-1:]))))]
            drafts.append(feed[0])

        h, cache = lm.layers(torch.cat(exits), exit_layer, lm.n_layers, cache)
        greedy = torch.argmax(lm.head(lm.norm(h[-(n_draft + 1) :])), dim=-1).tolist()
        n = 0
        while n < n_draft and drafts[n] == greedy[n]:
            n += 1
        seq += [*drafts[:n], greedy[n]]
        if len(seq) - len(prompt) == max_new_tokens:
            return seq[len(prompt) :]
