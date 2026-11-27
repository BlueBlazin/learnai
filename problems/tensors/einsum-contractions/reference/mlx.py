import mlx.core as mx


def attention_scores(q: mx.array, k: mx.array) -> mx.array:
    # out[b, h, t, s] = sum over d of q[b, h, t, d] * k[b, h, s, d]. The letters b and
    # h appear in both inputs and in the output: those axes are paired. d appears in
    # both inputs and not in the output: it is multiplied and summed.
    return mx.einsum("bhtd,bhsd->bhts", q, k)


def bilinear(x: mx.array, w: mx.array, y: mx.array) -> mx.array:
    # out[b] = sum over d and e of x[b, d] * w[d, e] * y[b, e]. The same b on x and y
    # pairs row b of x with row b of y.
    return mx.einsum("bd,de,be->b", x, w, y)


def per_head_project(x: mx.array, w: mx.array) -> mx.array:
    # out[b, h, t, e] = sum over d of x[b, h, t, d] * w[h, d, e]. The h on w is the h
    # on x, so head h is multiplied by its own matrix.
    return mx.einsum("bhtd,hde->bhte", x, w)
