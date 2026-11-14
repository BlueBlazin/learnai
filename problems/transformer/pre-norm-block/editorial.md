## The idea

A decoder is a **residual stream**: a tensor of shape `(B, T, D)` that starts as the
token embeddings and ends as the input of the output head. A block does not transform
the stream; it adds two corrections to it. Each correction is computed by a sublayer
from a normalized copy of the stream at that moment:

$$
h = x + \operatorname{Attn}(\operatorname{RMSNorm}_1(x)), \qquad
y = h + \operatorname{MLP}(\operatorname{RMSNorm}_2(h)) .
$$

Three properties follow, and each is a way to check an implementation.

1. **The stream is never normalized.** If both sublayers output zero, $y = x$ exactly,
   whatever the scale of $x$. One of the tests sets `o_proj` and `down_proj` to zero and
   checks precisely this. It also means the scale of the stream is free to grow from
   layer to layer, which is why a decoder ends with one more norm before the head.
2. **The sublayers are sequential.** The MLP reads $h$, which already contains the
   attention output. Information that attention moved to a position in this block is
   visible to the MLP of the same block.
3. **Each sublayer has its own norm.** The gains are trained separately and differ.

Pre-norm is preferred over the original post-norm layout,
$x \leftarrow \operatorname{Norm}(x + f(x))$, because the identity path from the first
layer to the last is never interrupted: gradients reach every layer directly, and deep
stacks train without the warm-up tricks post-norm needed. At inference time the only
thing that matters is that you run the layout the weights were trained with.

## Reference solution (PyTorch)

```python
def block(x, params, mask):
    h = ops.rms_norm(x, params["input_layernorm.weight"])
    q = h @ params["self_attn.q_proj.weight"].T
    k = h @ params["self_attn.k_proj.weight"].T
    v = h @ params["self_attn.v_proj.weight"].T
    x = x + ops.attention(q, k, v, mask) @ params["self_attn.o_proj.weight"].T

    h = ops.rms_norm(x, params["post_attention_layernorm.weight"])
    return x + ops.swiglu(
        h,
        params["mlp.gate_proj.weight"],
        params["mlp.up_proj.weight"],
        params["mlp.down_proj.weight"],
    )
```

The JAX and MLX versions are the same lines. The discipline that keeps the wiring
right is in the names: `x` is always the stream and is only reassigned as `x + ...`;
`h` is always a normalized copy and never survives its sublayer. Hugging Face writes
the same thing with a variable called `residual` that is saved before each norm and
added back after each sublayer.

## Common mistakes

- **Post-norm.** `x = norm(x + attn(x))` is the layout of the 2017 Transformer and of
  BERT. It uses the same parameters and gives the same shapes. With weights trained for
  pre-norm, the sublayers receive unnormalized inputs and the stream is rescaled twice
  per block.
- **Taking the second residual from the block input.** `return x_in + mlp(norm2(h))`
  computes the MLP correctly and then adds it to the wrong tensor: the attention output
  feeds the MLP but is missing from the result. This happens when the first sum is
  stored under a new name and the last line still says `x`.
- **Running the sublayers in parallel.** `h + mlp(norm2(x_in))` is the mirror image:
  the right tensor is added to, but the MLP never sees what attention wrote. (Some
  architectures are deliberately parallel; this one is not.)
- **One norm for both sublayers.** Reusing `input_layernorm.weight` before the MLP
  runs without error, because both gains have shape `(D,)`. In a freshly initialized
  model both gains are all ones and the bug is invisible, so a test with untrained
  weights does not catch it.
- **Normalizing the stream.** `x = norm(x); x = x + attn(x)` looks like pre-norm and
  feeds the sublayer the right input, but the residual now carries the normalized copy.
  The test with zeroed output projections returns `norm2(norm1(x))` instead of `x`.
- **Dropping the mask.** Calling the attention without the mask lets every position see
  the future. The output has the right shape, and at the last position of a causal
  sequence it is even correct, since that position may see everything anyway.

## Complexity

Attention costs $O(B\,T^2 D)$ time and $O(B\,T^2)$ memory for the weights; the
projections cost $O(B\,T\,D^2)$ and the MLP $O(B\,T\,D\,M)$. The norms and the two
additions are $O(B\,T\,D)$. For real models, where $M$ is several times $D$, the MLP
holds about two thirds of the block's parameters and of its arithmetic at short
context.

## In the wild

Hugging Face's `LlamaDecoderLayer.forward` is this function line for line: save
`residual`, apply `input_layernorm`, run `self_attn`, add; save `residual`, apply
`post_attention_layernorm`, run `mlp`, add. The parameter names in this problem are the
ones in those checkpoints, which is also where the slightly misleading name
`post_attention_layernorm` comes from: it is the norm *after the attention sublayer*,
which makes it the pre-norm of the MLP. It is not a post-norm.

Not every current model is purely pre-norm. Gemma 2 and Gemma 3 normalize the input
*and* the output of each sublayer before the addition, with four norms per block, and
the stream is still only added to. So the layout is a property of the model family
that has to be read from its reference implementation; the tensor shapes in the
checkpoint will not tell you.
