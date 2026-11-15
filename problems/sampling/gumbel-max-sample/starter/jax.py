import jax


def gumbel_sample(logits: jax.Array, uniforms: jax.Array) -> jax.Array:
    """Gumbel-max sampling: one token per row, argmax of logits plus Gumbel noise.

    logits: float32 tensor of shape (B, V), unnormalized; entries may be -inf (masked).
    uniforms: float32 tensor of shape (B, V), uniform numbers in [0, 1] supplied by the
        caller, one per logit; exactly 0 and exactly 1 both occur.
    Returns an integer tensor of shape (B,): for each row, the argmax over v of
    logits[b, v] - log(-log(u)), with u clamped to [1.1754944e-38, 0.99999994].
    Never the index of a masked token.
    """
    raise NotImplementedError
