import jax.numpy as jnp

from learnai.kit import TinyLM


def speculative_greedy(
    target: TinyLM,
    draft: TinyLM,
    prompt: list[int],
    k: int,
    max_new_tokens: int,
    eos_id: int | None,
) -> list[int]:
    seq = list(prompt)  # the verified sequence: the prompt, then every emitted token
    t_cache = d_cache = None
    t_len = d_len = 0  # how many tokens of `seq` each model's cache covers
    while True:
        # A round emits at most n_draft + 1 tokens, so never draft past the budget.
        n_draft = min(k, max_new_tokens - (len(seq) - len(prompt)) - 1)

        # Draft: propose tokens one at a time through the draft's cache. Each call
        # feeds what the cache does not cover yet (on the first call of a round, the
        # verified tokens the draft has not seen; then its own previous proposal).
        drafts: list[int] = []
        for _ in range(n_draft):
            logits, d_cache = draft((seq + drafts)[d_len:], d_cache)
            d_len = len(seq) + len(drafts)
            drafts.append(int(jnp.argmax(logits[-1])))

        # Verify: one target call over the unseen verified tokens and all drafts. Its
        # last n_draft + 1 rows score the draft positions and the one after them.
        logits, t_cache = target((seq + drafts)[t_len:], t_cache)
        greedy = jnp.argmax(logits[-(n_draft + 1) :], axis=-1).tolist()
        n = 0
        while n < n_draft and drafts[n] == greedy[n]:
            n += 1
        # The accepted prefix, then the target's own token: the correction at the
        # first mismatch, or the bonus token after a full match.
        new = [*drafts[:n], greedy[n]]
        if eos_id in new:
            new = new[: new.index(eos_id) + 1]
        seq += new
        if new[-1] == eos_id or len(seq) - len(prompt) == max_new_tokens:
            return seq[len(prompt) :]

        # Roll back: both caches may cover rejected drafts. Keep only verified
        # tokens, and never the newest one: it has to be fed next round, because its
        # logits are what the round starts from. (The draft's cache can be shorter
        # than that already: it never saw its own last proposal.)
        t_len = len(seq) - 1
        t_cache = t_cache.truncate(t_len)
        d_len = min(d_len, len(seq) - 1)
        d_cache = d_cache.truncate(d_len)
