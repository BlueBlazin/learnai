# BUG: bounds the live beams with the current length instead of max_new_tokens; with a positive length penalty a longer hypothesis can still score higher, so the search stops too early.
import torch

from whetstone.kit import TinyLM, ops


def beam_search(
    lm: TinyLM,
    prompt: list[int],
    n_beams: int,
    max_new_tokens: int,
    eos_id: int | None,
    length_penalty: float,
) -> list[tuple[list[int], float]]:
    logits, cache = lm(prompt)
    log_probs = ops.log_softmax(logits[-1:])  # (1, V)
    beams: list[list[int]] = [[]]
    scores = [0.0]
    finished: list[tuple[float, list[int]]] = []  # (final score, tokens), best first

    for step in range(1, max_new_tokens + 1):
        vocab = log_probs.shape[-1]
        totals = (torch.tensor(scores)[:, None] + log_probs).reshape(-1)
        order = torch.argsort(totals, descending=True, stable=True)[: 2 * n_beams]

        live: list[tuple[int, int, float]] = []  # (parent row, token, log-probability)
        for rank, (index, total) in enumerate(zip(order.tolist(), totals[order].tolist())):
            parent, token = divmod(index, vocab)
            if step == max_new_tokens or token == eos_id:
                if rank < n_beams:
                    final = total / step**length_penalty
                    finished.append((final, [*beams[parent], token]))
            elif len(live) < n_beams:
                live.append((parent, token, total))

        finished.sort(key=lambda hypothesis: -hypothesis[0])
        del finished[n_beams:]
        if step == max_new_tokens:
            break
        bound = live[0][2] / step**length_penalty
        if len(finished) == n_beams and bound <= finished[-1][0]:
            break

        parents = [parent for parent, _, _ in live]
        beams = [[*beams[parent], token] for parent, token, _ in live]
        scores = [total for _, _, total in live]
        logits, cache = lm([[token] for _, token, _ in live], cache.select(parents))
        log_probs = ops.log_softmax(logits[:, -1, :])  # (n_beams, V)

    return [(tokens, score) for score, tokens in finished]
