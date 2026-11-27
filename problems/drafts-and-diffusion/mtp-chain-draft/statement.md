Implement `mtp_chain_draft`, which drafts multiple tokens using a chained multi-token-prediction (MTP) module stored in a checkpoint under `mtp.*`. Unlike the parallel heads of `mtp-head-draft`, which all read the same hidden state, each stage in a chained MTP module runs a one-block transformer over the previous stage's hidden states and the token that stage just drafted.

## Specification

```python
def mtp_chain_draft(params, cfg, tokens, next_token, hidden, n_draft): ...  # -> (draft_tokens, draft_logits)
```

The main model has already read a prompt `tokens` of length $T$, containing token ids $t_0, \dots, t_{T-1}$ at positions $0$ to $T - 1$. The tensor `hidden` holds the main model's hidden states of width $D = \texttt{cfg.d\_model}$ **after** its final norm, so `hidden[i]` is the state at position $i$. From that prompt, the main model produced `next_token` $= t_T$ (a Python `int`). Your task is to run $n = \texttt{n\_draft}$ chained stages to draft the next $n$ tokens $t_{T+1}, \dots, t_{T+n}$ over a vocabulary of size $V = \texttt{cfg.vocab\_size}$.

The MTP module shares its token embedding matrix and output head with the main model:
- $E = \texttt{params["model.embed\_tokens.weight"]}$ of shape `(V, D)`.
- $W_\text{head} = E$ when `cfg.tie_embeddings` is `True`, otherwise `params["lm_head.weight"]` of shape `(V, D)`.

Let $\operatorname{rms}(x, g)$ denote `ops.rms_norm(x, g, cfg.norm_eps)` from `learnai.kit`, and let $[a\,; b]$ denote concatenation along the last axis.

### Stage computation

Set the initial hidden states to $h^0 = \texttt{hidden}$. Stages are indexed $j = 0, \dots, n - 1$, and stage $j$ uses its own weights under `mtp.layers.{j}.*`.

Each stage operates on $T$ rows. At stage $j$, the input token window is $t_{j+1}, \dots, t_{j+T}$ (the sequence shifted left by $j + 1$, ending with `next_token` at stage 0 and with the token drafted by stage $j - 1$ after that). Row $i$ ($0 \le i < T$) pairs the previous stage's hidden state $h^j_i$ with the embedding of the token one position ahead, $t_{i+j+1}$:

1. **Combine embedding and hidden state.** Normalize each half with `enorm.weight` and `hnorm.weight`, concatenate them with the embedding first (width $2D$), and project back to width $D$ with $W_\text{eh} = \texttt{eh\_proj.weight}$:
   $$
   x^j_i = \big[\operatorname{rms}(E[t_{i+j+1}],\ \texttt{enorm})\ ;\ \operatorname{rms}(h^j_i,\ \texttt{hnorm})\big]\; W_\text{eh}^\top
   $$
2. **Run one causal decoder block.** Pass all $T$ rows $x^j$ through $\operatorname{Block}_j$, using the weights under `mtp.layers.{j}.mtp_block.`:
   $$
   z^j = \operatorname{Block}_j(x^j)
   $$
3. **Apply the post-norm.** Normalize every row with `post_norm.weight` to produce the hidden states passed to stage $j + 1$:
   $$
   h^{j+1} = \operatorname{rms}(z^j,\ \texttt{post\_norm})
   $$
4. **Score and draft the next token.** Project only the last row $h^{j+1}_{T-1}$ through the shared output head and take its argmax:
   $$
   \ell^j = h^{j+1}_{T-1}\; W_\text{head}^\top, \qquad t_{T+1+j} = \operatorname{argmax}\ \ell^j
   $$

Stage $j + 1$ reads all $T$ rows of $h^{j+1}$, not `hidden`.

![Two chained stages. Stage 1 reads the states $h^1$ of stage 0 and embeds the token that stage 0 drafted.](assets/mtp-stages.svg)

### Decoder block

$\operatorname{Block}_j$ is the pre-norm transformer block from `learnai.kit`:

$$
a = x + \operatorname{Attn}\big(\operatorname{rms}(x, g_1)\big), \qquad
z = a + \operatorname{MLP}\big(\operatorname{rms}(a, g_2)\big)
$$

where $g_1$ is `input_layernorm.weight` and $g_2$ is `post_attention_layernorm.weight`.

- **Attention $\operatorname{Attn}(u)$:** Project $u$ with `q_proj` to `cfg.n_heads` heads of width `cfg.head_dim`, and with `k_proj` and `v_proj` to `cfg.n_kv_heads` heads. Apply `ops.rope(., positions, cfg.rope_base)` to queries and keys, not values. Call `ops.attention(q, k, v, mask)` with a causal mask (row $i$ attends to rows $\le i$). `ops.attention` handles grouped-query attention when `q` has shape `(1, n_heads, T, head_dim)` and `k`, `v` have shape `(1, n_kv_heads, T, head_dim)`. Merge the heads back to width $D$ and apply `o_proj`.
- **Positions:** For stage $j$, `positions` are $j + 1, \dots, j + T$, matching the tokens embedded by stage $j$. Because rotary attention scores depend only on position differences and there is no KV cache here, any $T$ consecutive positions produce the same result.
- **MLP $\operatorname{MLP}(u)$:** Compute `ops.swiglu(u, gate_proj, up_proj, down_proj)` with intermediate width $d_\text{mlp}$.

### Stage weights

All linear weights have shape `(out, in)` with no biases.

| key under `mtp.layers.{j}.` | shape |
|---|---|
| `enorm.weight`, `hnorm.weight`, `post_norm.weight` | `(D,)` |
| `eh_proj.weight` | `(D, 2D)` |
| `mtp_block.input_layernorm.weight` ($g_1$), `mtp_block.post_attention_layernorm.weight` ($g_2$) | `(D,)` |
| `mtp_block.self_attn.q_proj.weight`, `o_proj.weight` | `(D, D)` |
| `mtp_block.self_attn.k_proj.weight`, `v_proj.weight` | `(n_kv_heads * head_dim, D)` |
| `mtp_block.mlp.gate_proj.weight`, `up_proj.weight` | `(d_mlp, D)` |
| `mtp_block.mlp.down_proj.weight` | `(D, d_mlp)` |

Return `(draft_tokens, draft_logits)`, where `draft_tokens` holds $(t_{T+1}, \dots, t_{T+n})$ and `draft_logits` holds $(\ell^0, \dots, \ell^{n-1})$. Do not modify any input tensor, including entries of `params`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `tokens` | `(T,)` | integer |
| `hidden` | `(T, D)` | float32 |
| `draft_tokens` | `(n_draft,)` | integer |
| `draft_logits` | `(n_draft, V)` | float32 |

## Example

Using the default configuration and the kit's seeded weights with three stages:

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

- Stage 0 embeds `[10, 50, 1, 20, 2, 12, 9, 49]` (the prompt shifted left by one, followed by `next_token`) alongside the eight rows of `hidden`, at positions 1 to 8, and drafts `41`.
- Stage 1 embeds `[50, 1, 20, 2, 12, 9, 49, 41]` alongside stage 0's eight output rows, at positions 2 to 9, and drafts `25`.
- Stage 2 embeds `[1, 20, 2, 12, 9, 49, 41, 25]` alongside stage 1's eight output rows, at positions 3 to 10, and drafts `7`.

## Constraints

- $1 \le T \le 8$, $1 \le \texttt{n\_draft} \le 3$, and `params` contains at least `n_draft` stages.
- `d_model` $\le 32$, `vocab_size` $\le 64$, `n_heads` is a multiple of `n_kv_heads`, and `head_dim` is even.
- `next_token` need not be the main model's argmax.
- `cfg.norm_eps`, `cfg.rope_base`, the head counts, and `cfg.tie_embeddings` are not always the defaults.
- `draft_logits` is compared against `MtpModel` in Hugging Face Transformers and against the kit's oracle with relative tolerance `1e-3` and absolute tolerance `1e-4`.
- `draft_tokens` is compared exactly. In every test, the two largest logits of each stage differ by at least `0.05`.
- Do not call `mtp_draft_ref` or `decoder_block_ref` from `learnai.kit`.

## Why this matters

Production models such as Qwen3.5, GLM-5.3, and DeepSeek-V4.1 ship MTP stages directly inside their checkpoints. Each stage costs only one transformer block instead of a separate draft model, and the main model verifies the drafted tokens just as in `greedy-speculative-verify`.

A stage's weights expect one exact input layout: the hidden state of position $i$ paired with the embedding of token $i + 1$, with the embedding half first. If an implementation embeds the current token, swaps the two halves, or feeds every stage the main model's hidden states, it still drafts tokens without raising an error. Those drafts simply stop matching the main model, collapsing the acceptance rate and eliminating the speculative speedup.
