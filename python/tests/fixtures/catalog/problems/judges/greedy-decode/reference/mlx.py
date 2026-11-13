import mlx.core as mx


def greedy_decode(lm, prompt, max_new_tokens):
    logits, cache = lm(prompt)
    out = []
    while True:
        token = mx.argmax(logits[-1]).item()
        out.append(token)
        if len(out) == max_new_tokens:
            return out
        logits, cache = lm([token], cache)
