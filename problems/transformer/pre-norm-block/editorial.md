## The idea

A decoder maintains a **residual stream**, a tensor of shape `(B, T, D)` (batch size
$B$, sequence length $T$, model width $D$) that starts as the token embeddings and ends
as the input to the output head. A block does not replace the stream. Instead, it adds
two updates to it, where each sublayer reads a normalized copy of the current stream:

$$
h = x + \operatorname{Attn}(\operatorname{RMSNorm}_1(x)), \qquad
y = h + \operatorname{MLP}(\operatorname{RMSNorm}_2(h)) .
$$

Three properties follow from these equations, and each one gives you a way to check an
implementation.

1. **The stream is never normalized.** If both sublayers output zero, $y = x$ exactly,
   regardless of the scale of $x$. One of the tests sets `o_proj` and `down_proj` to
   zero to verify this directly. Because nothing rescales the stream inside the blocks,
   its magnitude can grow from layer to layer, which is why a decoder applies one final
   norm before the output head.
2. **The sublayers are sequential.** The MLP reads $h$, which already includes the
   attention output. Any information that attention moves into a position is immediately
   visible to the MLP in the same block.
3. **Each sublayer has its own norm.** The two gain vectors are trained separately and
   take different values.

Pre-norm is preferred over the original post-norm layout,
$x \leftarrow \operatorname{Norm}(x + f(x))$, because it preserves an uninterrupted
identity path from the first layer to the last. Gradients reach every layer directly, so
deep models train without the fragile warm-up schedules that post-norm required. At
inference time, what matters is matching the exact layout the weights were trained with.

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

The JAX and MLX versions are identical. Keeping variable names distinct helps prevent
wiring mistakes. Here `x` always holds the residual stream and is only updated as
`x + ...`, while `h` always holds a temporary normalized copy that is discarded after
its sublayer runs. Hugging Face uses the same pattern with a `residual` variable saved
before each norm and added back after each sublayer.

## Common mistakes

- **Post-norm.** Writing `x = norm(x + attn(x))` implements the 2017 Transformer and
  BERT layout using the same parameters and shapes. With weights trained for pre-norm,
  the sublayers receive unnormalized inputs and the stream is rescaled twice per block.
- **Adding the MLP output to the block input.** Writing `return x_in + mlp(norm2(h))`
  computes the MLP correctly but adds it to the wrong tensor, so the attention output
  feeds the MLP yet disappears from the block output. This commonly happens when the
  first sum is stored under a new name and the final line still references `x`.
- **Running the sublayers in parallel.** Writing `h + mlp(norm2(x_in))` adds to the
  right tensor, but the MLP never sees what attention wrote. Some architectures use
  parallel sublayers intentionally, but this layout is sequential.
- **Reusing one norm for both sublayers.** Using `input_layernorm.weight` before the
  MLP runs without error because both gains have shape `(D,)`. In a freshly initialized
  model both gains are all ones, so tests with untrained weights will not catch the bug.
- **Normalizing the stream.** Writing `x = norm(x); x = x + attn(x)` feeds the sublayer
  the right input, but overwrites the residual stream with the normalized copy. When
  output projections are zeroed, this returns `norm2(norm1(x))` instead of `x`.
- **Dropping the mask.** Calling attention without `mask` lets every position attend to
  future tokens. The output still has the right shape, and at the final position of a
  causal sequence it is even numerically right because that position can attend to
  everything anyway.

## Complexity

For batch size $B$, sequence length $T$, model width $D$, and MLP width $M$, attention
takes $O(B\,T^2 D)$ time and $O(B\,T^2)$ memory for the attention weights. The linear
projections take $O(B\,T\,D^2)$ time and the MLP takes $O(B\,T\,D\,M)$ time, while the
norms and residual additions take $O(B\,T\,D)$. In real models where $M$ is several
times $D$, the MLP accounts for roughly two thirds of the block's parameters and
short-context arithmetic.

## In the wild

Hugging Face's `LlamaDecoderLayer.forward` follows this structure line for line. It
saves `residual`, applies `input_layernorm`, runs `self_attn`, and adds the result back.
Then it saves `residual` again, applies `post_attention_layernorm`, runs `mlp`, and adds
that result back. The parameter names in this problem come from those checkpoints. That
is why `post_attention_layernorm` has a slightly misleading name. It sits after the
attention sublayer, which makes it the pre-norm for the MLP rather than a post-norm.

Not every modern model is purely pre-norm. Gemma 2 and Gemma 3 normalize both the input
and the output of each sublayer before adding to the residual stream, using four norms
per block while still leaving the stream untouched. Because tensor shapes in a
checkpoint cannot tell you the wiring, you always have to verify the layout against the
model family's reference implementation.
