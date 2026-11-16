import torch

from whetstone.kit import TinyLM, ops


class SpeculativeEngine:
    """Speculative decoding with a draft model and, optionally, prompt lookup.

    `target` and `draft` are kit models over one vocabulary:
        logits, cache = lm(tokens)          # tokens: the whole sequence so far
        logits, cache = lm(tokens, cache)   # tokens continue the cached prefix
        cache.truncate(n), cache.select(rows), cache.length
    """

    def __init__(
        self,
        target: TinyLM,
        draft: TinyLM,
        *,
        k: int = 4,
        adaptive: bool = False,
        lookup: tuple[int, int] | None = None,
    ):
        """`k`: draft length. `adaptive` (stage 3): change k after every round.
        `lookup` (stage 4): `(max_ngram, min_ngram)` turns prompt lookup on."""
        self.rounds: list[tuple[str, int, int]] = []
        raise NotImplementedError

    def generate(
        self, prompt: list[int], max_new_tokens: int, uniforms: torch.Tensor | None = None
    ) -> list[int]:
        """Generate `max_new_tokens` tokens after `prompt`; returns them as Python ints.

        Stage 1: `uniforms` is None and the result is the target's greedy continuation.
        Stage 2: `uniforms` is a float32 tensor of shape (R, J, 3) and the result is a
        sample from the target; round r uses `uniforms[r, j, 0]` to sample draft j,
        `uniforms[r, j, 1]` to test it, and `uniforms[r, n, 2]` for the round's last token.

        Afterwards `self.rounds` holds one `(proposer, n_drafted, n_accepted)` per round
        of this call, with proposer "draft" or "lookup".
        """
        raise NotImplementedError

    def generate_batch(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        """Stage 5. Greedy generation for prompts of equal length, one target call per round.

        Returns one list of `max_new_tokens` Python ints per prompt.
        """
        raise NotImplementedError
