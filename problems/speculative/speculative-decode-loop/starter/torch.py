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
    """Greedy speculative decoding: the target's greedy tokens in fewer target calls.

    target, draft: two models over one vocabulary. `logits, cache = lm(tokens)` runs a
        whole prefix; `logits, cache = lm(tokens, cache)` continues from a cache;
        `cache.truncate(n)` returns a cache covering only the first n tokens.
        `logits` has one row per token passed in that call, shape (T, vocab_size).
    prompt: the prompt token ids, at least one.
    k: the number of tokens the draft proposes per round, at least 1 (fewer when
        fewer than k + 1 tokens are left to generate).
    max_new_tokens: the maximum number of tokens to generate, at least 1.
    eos_id: generation stops after this token is generated; None disables stopping.
    Per round: the draft proposes tokens through its cache, one target call scores
    all of them, the agreeing prefix plus the target's own next token is kept, and
    both caches are truncated to verified tokens.
    Returns the generated tokens (without the prompt) as a list of Python ints: the
    target's greedy decode, ending with eos_id if it was generated.
    """
    raise NotImplementedError
