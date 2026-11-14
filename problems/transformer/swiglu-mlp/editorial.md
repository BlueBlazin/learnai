## The idea

A classic transformer MLP is $W_2\,\operatorname{act}(W_1 x)$: expand, apply a fixed
nonlinearity, contract. A gated linear unit replaces the fixed nonlinearity by a
multiplication between two projections of the same input, one of which has been passed
through an activation:

$$
y = W_{\text{down}} \big( \operatorname{act}(W_{\text{gate}}\,x) \odot W_{\text{up}}\,x \big).
$$

Each hidden unit computes a linear feature $u_i$ and scales it by a factor
$\operatorname{act}(g_i)$ that depends on the input through a second linear feature.
Shazeer (2020) compared the choices of activation inside a transformer: the sigmoid
gives the original GLU, GELU gives GeGLU, and Swish with $\beta = 1$, which is SiLU,
$z\,\sigma(z)$, gives **SwiGLU**. In matrix form for a batch of row vectors, with the
`(out, in)` layout:

$$
Y = \big(\operatorname{silu}(X W_{\text{gate}}^\top) \odot (X W_{\text{up}}^\top)\big)\, W_{\text{down}}^\top .
$$

There is nothing else to it, and that is the difficulty: the function has no
invariant that a wrong version breaks loudly. Every variant below returns the right
shape, and the numbers are of the right size. Only a comparison with the exact formula
tells them apart.

## Reference solution (PyTorch)

```python
def swiglu(x, w_gate, w_up, w_down):
    gate = x @ w_gate.T
    up = x @ w_up.T
    hidden = F.silu(gate) * up
    return hidden @ w_down.T
```

`x @ w.T` broadcasts over any leading axes, so a single vector, a `(B, d)` batch, and a
`(B, T, d)` batch need no special handling. In JAX the activation is `jax.nn.silu`, in
MLX `mlx.nn.silu`; the rest is identical.

The built-in SiLU is also the numerically safe one. $z\,\sigma(z)$ is fine for any
finite $z$: for $z = -100$ the sigmoid underflows to 0 and the product is 0, which is
the correctly rounded answer, and for $z = 100$ the sigmoid is exactly 1. A hand-written
$z\,e^{z} / (1 + e^{z})$ overflows to `inf / inf` for $z > 88.7$ in float32.

## Common mistakes

- **Activation on the wrong branch.** `gate * silu(up)` is the same function with the
  roles of the two matrices exchanged. With your own random weights it trains equally
  well, which is why it survives in code written from memory. With a checkpoint it is
  wrong, because `gate_proj` and `up_proj` were trained in their roles.
- **GELU in place of SiLU.** The two curves almost coincide: both pass through 0, dip
  slightly below it for negative inputs, and approach the identity. They differ by
  up to about 0.2 around $|z| \approx 2$. Gemma gates with GELU (GeGLU), in the tanh
  approximation; Llama, Qwen, and Granite gate with SiLU. Read the `hidden_act` field
  of the config.
- **Sigmoid in place of SiLU.** $\sigma(g) \odot u$ is the original GLU of Dauphin et
  al. (2017). The name "gate" suggests a value between 0 and 1, but the SwiGLU gate
  $g\,\sigma(g)$ is unbounded above and slightly negative below zero.
- **Activation after the product.** $\operatorname{silu}(g \odot u)$ looks like "apply
  the activation to the hidden layer", the habit from two-matrix MLPs.
- **Not transposing the weights.** `x @ w_gate` raises a shape error as soon as
  $h \ne d$, which is always the case in a real model ($h$ is about $\tfrac{8}{3} d$
  or more). In a toy with $d = h = d_{out}$ it runs and applies the transposed maps.
  Test with widths that are all different, and then with widths that are all equal.

## Complexity

Three matrix products: $O(N\,h\,(2d + d_{out}))$ time for $N$ input vectors, and
$O(N\,h)$ extra memory for the two branches. With $d_{out} = d$ the layer has $3\,d\,h$
parameters.

## In the wild

Meta's Llama code writes the layer as `w2(F.silu(w1(x)) * w3(x))`; Hugging Face
Transformers names the matrices and writes
`down_proj(act_fn(gate_proj(x)) * up_proj(x))`. Both are the formula of this problem.

The hidden width is chosen so that the parameter count matches the classic MLP. A
two-matrix MLP with $h = 4d$ has $8d^2$ parameters; three matrices with
$h = \tfrac{2}{3} \cdot 4d$ have the same $8d^2$. That is where widths such as 11008
for $d = 4096$ (Llama 2 7B: $\tfrac{8}{3} \cdot 4096$ rounded up to a multiple of 256)
come from. Newer models choose $h$ freely: Llama 3 8B uses 14336 for $d = 4096$, and
Qwen3-0.6B uses 3072 for $d = 1024$.

Because the gate and up projections read the same input, some models store them as
one fused matrix of shape `(2h, d)` and split the result in two (Phi-3's
`gate_up_proj`), which turns two matrix products into one larger one. If you load such
a checkpoint, the order of the two halves is one more convention to get right.
