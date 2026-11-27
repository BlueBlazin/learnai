Inspect the weights before writing code. Print the mean of `input_layernorm.weight`, `post_attention_layernorm.weight`, and `linear_attn.norm.weight` for your layer. A plain multiplicative gain should sit near `1.0`. Check which of the three already does, and what offset brings the other two near `1.0`.

---

The block structure is four steps: norm, mixer, residual add, then norm, MLP, residual add. Both block norms use `ops.rms_norm(u, 1 + w, 1e-6)`. Pass `1e-6` explicitly every time because `ops.rms_norm` defaults to `1e-5`, which disappears on unit-variance inputs but shifts real `0.03`-scale hidden states by about 0.5%.

The mixer is the same as `gated-delta-layer` with three adaptations:
- Read its nine tensors under `p + "linear_attn."`, keeping the trailing dot in `p` so layer 1 does not match layers 10 through 19.
- Derive sizes from shapes: `H = A_log.shape[0]`, `dv = norm.weight.shape[0]`, and `dk = (in_proj_qkv.shape[0] - H * dv) // (2 * H)`.
- Key heads and value heads have the same count, so no head repetition is needed, and `linear_attn.norm.weight` is passed to `ops.rms_norm` as stored (without adding `1`).

---

If one test passes and another fails, narrow down the bug by symptom:

- **Every entry is wrong, starting at row 0 of a new sequence:** check the norm conventions (missing `1 +` on a block norm or extra `1 +` on the gated norm), the convolution kernel direction, or whether you applied the gate before the norm.
- **Row 0 is right, but all later rows are wrong:** check the decay formula `exp(-exp(A_log) * softplus(a + dt_bias))`. On row 0 of a new sequence, `S` is all zeros, so decay has no effect there.
- **One full call is right, but split calls fail after the boundary:** check state propagation. The first `K - 1` rows after a split rely on `conv_state` (unactivated rows of `P`), and all subsequent rows rely on `S` with shape `(H, dv, dk)`. Make sure you do not mutate `S` in place.
- **The small model and real layer 13 pass, but real layer 1 fails:** check your layer-name prefix for a missing trailing dot.
