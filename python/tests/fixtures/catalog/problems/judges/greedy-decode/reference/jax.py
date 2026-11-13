import jax.numpy as jnp


def greedy_decode(lm, prompt, max_new_tokens):
    logits, cache = lm(prompt)
    out = []
    while True:
        token = int(jnp.argmax(logits[-1]))
        out.append(token)
        if len(out) == max_new_tokens:
            return out
        logits, cache = lm([token], cache)
