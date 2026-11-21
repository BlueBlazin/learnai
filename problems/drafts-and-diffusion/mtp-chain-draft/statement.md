Draft several tokens with the multi-token-prediction (MTP) module that ships inside a
checkpoint under `mtp.*`: a chain of one-block stages on top of the main model's final
hidden states. Each stage reads the stage before it and the token that stage drafted;
the parallel heads of `mtp-head-draft` all read the same hidden state.

## Specification

```python
def mtp_chain_draft(params, cfg, tokens, next_token, hidden, n_draft): ...  # -> (tokens, logits)
```

The main model has read the tokens $t_0, \dots, t_{T-1}$ at positions $0$ to $T - 1$;
`hidden[i]` is its hidden state at position $i$ **after** its final norm, and it has
produced `next_token` $= t_T$, a Python int. The module drafts $t_{T+1}, \dots, t_{T + n}$ with
$n$ = `n_draft`, one token per stage.

Stage $j = 0, \dots, n - 1$ has its own weights `mtp.layers.{j}.*` and works on $T$
rows. Row $i$ pairs a hidden state for position $i + j$ with the embedding of the token
one further, $t_{i+j+1}$:

$$
\begin{aligned}
x^j_i &= \big[\operatorname{rms}(E[t_{i+j+1}],\ \texttt{enorm})\ ;\ \operatorname{rms}(h^j_i,\ \texttt{hnorm})\big]\; W_\text{eh}^\top
  && \text{embedding first, then hidden state} \\
z^j &= \operatorname{Block}_j(x^j) && \text{one causal decoder block} \\
h^{j+1} &= \operatorname{rms}(z^j,\ \texttt{post\_norm}) \\
\ell^j &= h^{j+1}_{T-1}\; W_\text{head}^\top, \qquad t_{T+1+j} = \operatorname{argmax}\ \ell^j
  && \text{last row only}
\end{aligned}
$$

- $h^0$ is `hidden`. Stage $j + 1$ reads $h^{j+1}$, all $T$ rows of it, not `hidden`.
- The ids of stage $j$ are $t_{j+1}, \dots, t_{j+T}$: the sequence shifted left by
  $j + 1$, ending with `next_token` for stage 0 and with the token drafted by stage
  $j - 1$ after that.
- $E$ is `params["model.embed_tokens.weight"]`, and $W_\text{head}$ is the main model's
  head: $E$ when `cfg.tie_embeddings`, else `params["lm_head.weight"]`. The module has
  neither of its own.
- $\operatorname{rms}(x, g)$ is `ops.rms_norm(x, g, cfg.norm_eps)` from `learnai.kit`,
  and $[a\,;b]$ concatenates along the last axis, giving width $2D$.

$\operatorname{Block}_j$ is the pre-norm block of the kit decoder, with the weights
under `mtp.layers.{j}.mtp_block.`:

$$
a = x + \operatorname{Attn}\big(\operatorname{rms}(x, g_1)\big), \qquad
z = a + \operatorname{MLP}\big(\operatorname{rms}(a, g_2)\big).
$$

- $\operatorname{Attn}(u)$: project with `q_proj` (to `n_heads` heads of `head_dim`)
  and with `k_proj`, `v_proj` (to `n_kv_heads` heads); rotate queries and keys, not
  values, with `ops.rope(., positions, cfg.rope_base)`; `ops.attention(q, k, v, mask)`
  with a causal mask (row $i$ sees rows $\le i$), which handles the grouped heads when
  `q` is `(1, n_heads, T, head_dim)` and `k`, `v` are `(1, n_kv_heads, T, head_dim)`;
  merge the heads back to width $D$ and apply `o_proj`.
- `positions` for stage $j$ are $j + 1, \dots, j + T$, the positions of the tokens it
  embeds. Rotary scores depend only on differences of positions, so in this form,
  without a cache, any $T$ consecutive positions give the same result and the tests
  cannot tell.
- $\operatorname{MLP}(u)$ is `ops.swiglu(u, gate_proj, up_proj, down_proj)`.

| under `mtp.layers.{j}.` | shape |
|---|---|
| `enorm.weight`, `hnorm.weight`, `post_norm.weight` | `(D,)` |
| `eh_proj.weight` | `(D, 2D)` |
| `mtp_block.input_layernorm.weight` ($g_1$), `mtp_block.post_attention_layernorm.weight` ($g_2$) | `(D,)` |
| `mtp_block.self_attn.q_proj.weight`, `o_proj.weight` | `(D, D)` |
| `mtp_block.self_attn.k_proj.weight`, `v_proj.weight` | `(n_kv_heads * head_dim, D)` |
| `mtp_block.mlp.gate_proj.weight`, `up_proj.weight`; `down_proj.weight` | `(d_mlp, D)`; `(D, d_mlp)` |

Linear weights are `(out, in)`, without biases; $D$ is `cfg.d_model`. The function
returns `draft_tokens` $= (t_{T+1}, \dots, t_{T+n})$ and `draft_logits`
$= (\ell^0, \dots, \ell^{n-1})$. No tensor passed in is modified, `params` included.

## Shapes

| name | shape | dtype |
|---|---|---|
| `tokens` | `(T,)` | integer |
| `hidden` | `(T, D)` | float32 |
| `draft_tokens` | `(n_draft,)` | integer |
| `draft_logits` | `(n_draft, V)` | float32 |

## Example

The default configuration and the kit's seeded weights with three stages:

```python
from learnai.kit import DecoderConfig, decoder_hidden_ref, mtp_params

cfg = DecoderConfig(n_layers=2)  # V = 64, D = 32, 4 heads, 2 KV heads, tied head
params = mtp_params(cfg, n_layers=3, seed=8)
tokens = [9, 10, 50, 1, 20, 2, 12, 9]
hidden = decoder_hidden_ref(cfg, params, tokens)  # (8, 32)

draft_tokens, draft_logits = mtp_chain_draft(params, cfg, tokens, 49, hidden, 3)
# draft_tokens = [41, 25, 7]
# draft_logits[0][41], [1][25], [2][7] = 5.890466, 5.082431, 4.953575
```

Stage 0 embeds `[10, 50, 1, 20, 2, 12, 9, 49]` (the prompt shifted by one, then
`next_token`) next to the eight rows of `hidden`, at positions 1 to 8, and drafts 41.
Stage 1 embeds `[50, 1, 20, 2, 12, 9, 49, 41]` next to stage 0's eight output rows, at
positions 2 to 9, and drafts 25. Stage 2 embeds `[1, 20, 2, 12, 9, 49, 41, 25]` and
drafts 7.

## Constraints

- $1 \le T \le 8$, $1 \le$ `n_draft` $\le 3$, and `params` holds at least `n_draft`
  stages; `d_model` $\le 32$, `vocab_size` $\le 64$; `n_heads` is a multiple of
  `n_kv_heads` and `head_dim` is even. `next_token` need not be the main model's
  argmax.
- `cfg.norm_eps`, `cfg.rope_base`, the head counts, and `cfg.tie_embeddings` are not
  always the defaults.
- `draft_logits` is compared with the output of `MtpModel` in Hugging Face
  Transformers and with the kit's oracle: relative tolerance `1e-3`, absolute tolerance
  `1e-4`. `draft_tokens` is compared exactly; in every test the two largest logits of
  each stage differ by at least 0.05.
- Do not call the kit's `mtp_draft_ref` or `decoder_block_ref`.

## Why this matters

Qwen3.5 checkpoints carry one such stage, GLM-5.3 declares one and DeepSeek-V4.1
three. They cost one block each instead of a second model, and their drafts are
verified by the main model exactly as in `greedy-speculative-verify`. A stage's weights
are for one specific input: the hidden state of position $i$ next to the embedding of
token $i + 1$, embedding first. A loader that embeds the current token, swaps the two
halves, or feeds every stage the main model's states still drafts tokens without any
error; they just stop matching what the main model produces, and the speedup goes
with the acceptance rate.
