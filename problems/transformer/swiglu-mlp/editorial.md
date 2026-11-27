## The idea

A classic transformer MLP computes $W_2\,\operatorname{act}(W_1 x)$, which expands the
input, applies a fixed nonlinearity, and projects back down. A gated linear unit
replaces that fixed nonlinearity with the elementwise product of two projections of the
same input, where only one projection passes through an activation:

$$
y = W_{\text{down}} \big( \operatorname{act}(W_{\text{gate}}\,x) \odot W_{\text{up}}\,x \big).
$$

Each hidden unit $i$ computes a linear feature $u_i = (W_{\text{up}}\,x)_i$ and scales
it by an input-dependent gate $\operatorname{act}(g_i)$, where
$g_i = (W_{\text{gate}}\,x)_i$. Shazeer (2020) compared activation choices inside a
transformer. Using sigmoid gives the original GLU, using GELU gives GeGLU, and using
Swish with $\beta = 1$ (which is SiLU, $z\,\sigma(z)$) gives **SwiGLU**.

For a batch of row vectors $X$ and weights in `(out, in)` layout, each linear
projection multiplies on the right by the transposed weight matrix, giving output $Y$:

$$
Y = \big(\operatorname{silu}(X W_{\text{gate}}^\top) \odot (X W_{\text{up}}^\top)\big)\, W_{\text{down}}^\top .
$$

Wrong variants of this formula rarely fail loudly. Every bug below returns the right
shape and produces values of a reasonable scale, so only comparing against the exact
formula separates them.

## Reference solution (PyTorch)

```python
def swiglu(x, w_gate, w_up, w_down):
    gate = x @ w_gate.T
    up = x @ w_up.T
    hidden = F.silu(gate) * up
    return hidden @ w_down.T
```

The matrix product `x @ w.T` broadcasts over any leading axes, so a single `(d,)`
vector, a `(B, d)` batch of vectors, and a `(B, T, d)` batch of sequences all work
without special cases. In JAX the
activation is `jax.nn.silu`, and in MLX it is `mlx.nn.silu`. The rest of the code is
identical.

The built-in SiLU is also numerically safe across the full float32 range. Computing
$z\,\sigma(z)$ stays finite for any finite $z$. At $z = -100$ the sigmoid underflows to
0 and the product is 0, and at $z = 100$ the sigmoid is 1. By contrast, a hand-written
$z\,e^{z} / (1 + e^{z})$ overflows to `inf / inf` (NaN) for $z > 88.7$ in float32.

## Common mistakes

- **Activation on the wrong branch.** Computing `gate * silu(up)` swaps the roles of
  `w_gate` and `w_up`. Trained from scratch with random weights, it learns equally well,
  so the bug can go unnoticed in fresh code. Loaded with a trained checkpoint, it breaks
  because `gate_proj` and `up_proj` were trained in their assigned roles.
- **GELU in place of SiLU.** Both curves pass through 0, dip slightly below zero for
  negative inputs, and approach the identity for large positive inputs. They still
  differ by up to about 0.2 near $|z| \approx 2$. Gemma uses GELU gating (GeGLU with the
  tanh approximation), whereas Llama, Qwen, and Granite use SiLU. Always check the
  `hidden_act` field in the model config.
- **Sigmoid in place of SiLU.** Computing $\sigma(g) \odot u$ gives the original GLU
  from Dauphin et al. (2017). Although it is called a gate, the SwiGLU factor
  $g\,\sigma(g)$ is unbounded above and slightly negative below zero rather than
  bounded in $[0, 1]$.
- **Activation after the product.** Writing $\operatorname{silu}(g \odot u)$ applies
  the nonlinearity after combining the branches, a habit carried over from two-matrix
  MLPs.
- **Not transposing the weights.** Writing `x @ w_gate` raises a shape error whenever
  $h \ne d$, which is always true in real models where $h \approx \tfrac{8}{3} d$ or
  larger. In a small test where $d = h = d_{\text{out}}$, it runs silently and applies
  the transposed matrices. Test with all three widths distinct, and also with all three
  equal.

## Complexity

For $N$ total input vectors, the three matrix products take $O(N\,h\,(2d + d_{\text{out}}))$
time and $O(N\,h)$ extra memory for the two hidden branches. When $d_{\text{out}} = d$,
the layer has $3\,d\,h$ parameters.

## In the wild

Meta's Llama implementation writes `w2(F.silu(w1(x)) * w3(x))`. Hugging Face
Transformers names the matrices explicitly and writes
`down_proj(act_fn(gate_proj(x)) * up_proj(x))`. Both compute the exact formula in this
problem.

Model authors choose $h$ to match the parameter count of a classic MLP. A two-matrix
MLP with $h = 4d$ has $8d^2$ parameters, and three matrices with
$h = \tfrac{2}{3} \cdot 4d = \tfrac{8}{3} d$ have the same $8d^2$ parameters. That is
why Llama 2 7B uses $h = 11008$ for $d = 4096$ ($\tfrac{8}{3} \cdot 4096$ rounded up to
a multiple of 256). Newer models choose $h$ more freely. Llama 3 8B uses 14336 for
$d = 4096$, and Qwen3-0.6B uses 3072 for $d = 1024$.

Because `w_gate` and `w_up` read the same input, some models pack them into a single
fused matrix of shape `(2h, d)` (such as Phi-3's `gate_up_proj`) and split the output
into two halves. That merges two matrix products into one larger product, while adding
one more convention to verify when loading a checkpoint: which half is the gate.
