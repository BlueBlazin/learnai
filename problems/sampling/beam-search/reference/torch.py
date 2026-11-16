import torch

from learnai.kit import TinyLM, ops


def beam_search(
    lm: TinyLM,
    prompt: list[int],
    n_beams: int,
    max_new_tokens: int,
    eos_id: int | None,
    length_penalty: float,
) -> list[tuple[list[int], float]]:
    # Prefill. There is one live hypothesis, the empty one, with log-probability 0.
    logits, cache = lm(prompt)
    log_probs = ops.log_softmax(logits[-1:])  # (1, V)
    beams: list[list[int]] = [[]]
    scores = [0.0]
    finished: list[tuple[float, list[int]]] = []  # (final score, tokens), best first

    for step in range(1, max_new_tokens + 1):
        # Candidate (beam i, token v) sits at index i * V + v of the flattened totals.
        vocab = log_probs.shape[-1]
        totals = (torch.tensor(scores)[:, None] + log_probs).reshape(-1)
        # A stable sort: equal totals stay in index order (lower beam, lower token).
        order = torch.argsort(totals, descending=True, stable=True)[: 2 * n_beams]

        live: list[tuple[int, int, float]] = []  # (parent row, token, log-probability)
        for rank, (index, total) in enumerate(zip(order.tolist(), totals[order].tolist())):
            parent, token = divmod(index, vocab)
            if step == max_new_tokens or token == eos_id:
                # Only a candidate that would have survived pruning may finish.
                if rank < n_beams:
                    final = total / step**length_penalty
                    finished.append((final, [*beams[parent], token]))
            elif len(live) < n_beams:
                live.append((parent, token, total))

        # Keep the n_beams best finished hypotheses. The sort is stable, so of two
        # equal scores the one that finished earlier stays ahead.
        finished.sort(key=lambda hypothesis: -hypothesis[0])
        del finished[n_beams:]
        if step == max_new_tokens:
            break
        # Log-probabilities only fall and a hypothesis has at most max_new_tokens
        # tokens, so no descendant of a live beam can score above this bound.
        bound = live[0][2] / max_new_tokens**length_penalty
        if len(finished) == n_beams and bound <= finished[-1][0]:
            break

        # One batched call for all live beams. Row j of the new batch continues row
        # parents[j] of the old one, so the cache rows are gathered in that order.
        parents = [parent for parent, _, _ in live]
        beams = [[*beams[parent], token] for parent, token, _ in live]
        scores = [total for _, _, total in live]
        logits, cache = lm([[token] for _, token, _ in live], cache.select(parents))
        log_probs = ops.log_softmax(logits[:, -1, :])  # (n_beams, V)

    return [(tokens, score) for score, tokens in finished]
