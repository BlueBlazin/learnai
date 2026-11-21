import mlx.core as mx


def ple_inputs(
    tokens: mx.array,
    h0: mx.array,
    table: mx.array,
    w_model: mx.array,
    norm_gain: mx.array,
    n_layers: int,
) -> mx.array:
    """The per-layer embeddings of T tokens, for every layer at once.

    tokens: integer tensor, shape (T,). Token ids.
    h0: float32, shape (T, D). The input of layer 0 (the scaled token embeddings).
    table: float32, shape (V, n_layers * d_ple). One packed row per token.
    w_model: float32, shape (n_layers * d_ple, D). Projection of h0, no bias.
    norm_gain: float32, shape (d_ple,). Gain of the norm on the projected part.
    n_layers: number of layers L.
    Returns float32, shape (T, L, d_ple); [:, i] is what layer i receives.
    """
    raise NotImplementedError


def ple_inject(
    h: mx.array,
    ple_i: mx.array,
    w_gate: mx.array,
    w_proj: mx.array,
    post_gain: mx.array,
) -> mx.array:
    """Add one layer's per-layer embedding to the stream.

    h: float32, shape (T, D). The stream.
    ple_i: float32, shape (T, d_ple). This layer's slice of the per-layer embeddings.
    w_gate: float32, shape (d_ple, D). w_proj: float32, shape (D, d_ple). No biases.
    post_gain: float32, shape (D,). Gain of the norm on the projected branch.
    Returns float32, shape (T, D).
    """
    raise NotImplementedError


def ple_layer(
    params: dict[str, mx.array],
    x: mx.array,
    ple: mx.array,
    i: int,
    mask: mx.array,
) -> mx.array:
    """One decoder layer that receives a per-layer embedding.

    params: the layer's weights by name (see the statement).
    x: float32, shape (T, D). The stream entering the layer.
    ple: float32, shape (T, L, d_ple). The result of ple_inputs.
    i: index of this layer, 0 <= i < L.
    mask: bool, shape (T, T). True where a query row may attend to a key column.
    Returns float32, shape (T, D).
    """
    raise NotImplementedError
