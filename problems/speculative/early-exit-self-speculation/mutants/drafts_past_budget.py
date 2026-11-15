# BUG: always drafts k tokens, also when fewer than k + 1 tokens are left to generate, and cuts the result to max_new_tokens afterwards: layers run on drafts that can never be emitted.
import torch

from whetstone.kit import LayeredLM


def self_speculative_greedy(
    lm: LayeredLM, prompt: list[int], exit_layer: int, k: int, max_new_tokens: int
) -> list[int]:
    seq = list(prompt)  # the verified sequence: the prompt, then every emitted token
    cache = lm.empty_cache()  # one cache for all layers; early and late advance together
    while True:
        seen = cache.length  # every layer has processed seq[:seen]
        # A round emits at most n_draft + 1 tokens, so never draft past the budget.
        n_draft = k

        # Draft with layers 0 .. E - 1. The first call feeds the verified tokens the
        # layers have not seen (the whole prompt in round one, afterwards the newest
        # token); each later call feeds the previous draft. Every exit state is kept:
        # it is the input the late layers need, so the early layers never run twice.
        feed, position = seq[seen:], seen
        exits, drafts = [], []
        while True:
            h, cache = lm.layers(lm.embed(feed, offset=position), 0, exit_layer, cache)
            exits.append(h)
            position += len(feed)
            if len(drafts) == n_draft:
                break  # the last draft was fed for its exit state only
            # Early exit: the shared head on the *normalized* state of the last position.
            feed = [int(torch.argmax(lm.head(lm.norm(h[-1:]))))]
            drafts.append(feed[0])

        # Verify with layers E .. L - 1, once, on all saved exit states. The last
        # n_draft + 1 rows are the positions of the newest verified token and the drafts.
        h, cache = lm.layers(torch.cat(exits), exit_layer, lm.n_layers, cache)
        greedy = torch.argmax(lm.head(lm.norm(h[-(n_draft + 1) :])), dim=-1).tolist()
        n = 0
        while n < n_draft and drafts[n] == greedy[n]:
            n += 1
        # The accepted prefix, then the full model's own token: the correction at the
        # first mismatch, or the bonus token after a full match.
        seq += [*drafts[:n], greedy[n]]
        if len(seq) - len(prompt) >= max_new_tokens:
            return seq[len(prompt) : len(prompt) + max_new_tokens]

        # Roll back: every layer may hold rejected drafts. Keep verified tokens only,
        # and not the newest one, which no layer has processed yet.
        cache = cache.truncate(len(seq) - 1)
