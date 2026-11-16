import jax.numpy as jnp

from learnai.kit import LayeredLM


def self_speculative_greedy(
    lm: LayeredLM, prompt: list[int], exit_layer: int, k: int, max_new_tokens: int
) -> list[int]:
    """Greedy decoding in which the model's own early layers draft for its late layers.

    lm: a LayeredLM with L = lm.n_layers layers.
        h = lm.embed(tokens, offset=n)        hidden states of tokens at positions n, n + 1, ...
        h, cache = lm.layers(h, a, b, cache)  layers a .. b - 1, continuing what they cached
        logits = lm.head(lm.norm(h))          the shared output head, after the final norm
        cache = lm.empty_cache(); cache.truncate(n) rolls every layer back to n tokens.
    prompt: the prompt token ids, at least one.
    exit_layer: E with 1 <= E < L; drafts come from the states after layer E - 1.
    k: the number of tokens drafted per round, at least 1 (fewer when fewer than k + 1
        tokens are left to generate).
    max_new_tokens: the number of tokens to generate, at least 1.
    Per round: draft through layers 0 .. E - 1 and the head, keeping every exit state;
    run layers E .. L - 1 once on those states; keep the agreeing prefix plus the full
    model's own next token; roll the cache back to verified tokens.
    Returns the generated tokens (without the prompt) as a list of Python ints: the
    full model's greedy decode.
    """
    raise NotImplementedError
