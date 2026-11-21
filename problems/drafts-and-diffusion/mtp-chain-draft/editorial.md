## The idea

A draft model has to be cheap and has to agree with the main model. An MTP module gets
both by starting from what the main model already computed. The main model's last
hidden state at position $i$ is a summary of everything up to $i$, good enough that one
matrix product with the head predicts token $i + 1$. One more block on top of it, told
which token $i + 1$ actually was, can predict token $i + 2$:

- stage 0 pairs the hidden state of position $i$ with the embedding of token $i + 1$
  and produces a state that predicts token $i + 2$;
- stage 1 pairs *that* state with the embedding of token $i + 2$ and predicts token
  $i + 3$; and so on.

That is a chain. Each stage's state is "what the main model would know one position
later", approximated by one block instead of the full stack, and the token it needs
is the one the previous stage just drafted. Compare `mtp-head-draft`: there, every head
reads the same hidden state and no head knows what the others guessed, so head 2 has to
predict token $i + 3$ without knowing tokens $i + 1$ and $i + 2$. The chain conditions
each draft on the earlier ones; the price is that the stages run one after another.

The embedding matrix and the output head are the main model's, so a stage adds only
two norms, one projection of width $2D \to D$, one block, and a post norm.

## The bookkeeping

Two things move from stage to stage, and each classic mistake breaks one of them.

- **The window of ids** has $T$ entries and runs one token ahead of the hidden states:
  `tokens[1:] + [next_token]` for stage 0. After every stage it rolls: drop the first
  id, append the draft.
- **The hidden states** are `hidden` for stage 0 and then the previous stage's whole
  output, after its post norm. All $T$ rows matter: the block is causal attention over
  the rows, so row $T - 1$ of stage 1 attends to rows $0..T-1$ of stage 0's output.
  That is also why the causal mask matters although only the last row is decoded: row
  $T - 1$ of stage 0 sees everything with or without a mask, but its other rows change,
  and stage 1 reads them.

Only the last row's logits are computed, as in the library (it computes all rows only
when it is given labels for a loss).

## Positions

Stage $j$'s rows hold tokens $j + 1$ to $j + T$, and the library gives them those
positions (it adds 1 to every position after each stage). In this problem a stage
attends only among its own $T$ rows, and rotary attention scores depend on position
differences alone, so shifting all positions of a stage by a constant changes nothing:
measured, positions $0..T-1$ in every stage reproduce the logits to $3 \cdot 10^{-6}$.
The offset becomes visible as soon as a stage keeps a key-value cache across decoding
rounds, as the library's generator does: cached keys were rotated with their absolute
positions, and new queries must be rotated consistently.

## Reference solution (PyTorch)

```python
def mtp_chain_draft(params, cfg, tokens, next_token, hidden, n_draft):
    seq = tokens.shape[0]
    embedding = params["model.embed_tokens.weight"]
    head = embedding if cfg.tie_embeddings else params["lm_head.weight"]
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))

    def heads(t, n):  # (T, n * head_dim) -> (1, n, T, head_dim)
        return t.reshape(seq, n, cfg.head_dim).transpose(0, 1)[None]

    ids = torch.cat((tokens[1:], torch.tensor([next_token], dtype=tokens.dtype)))
    h = hidden
    drafted, logits = [], []
    for j in range(n_draft):
        p = f"mtp.layers.{j}."
        e = ops.rms_norm(embedding[ids], params[p + "enorm.weight"], cfg.norm_eps)
        g = ops.rms_norm(h, params[p + "hnorm.weight"], cfg.norm_eps)
        x = torch.cat((e, g), dim=-1) @ params[p + "eh_proj.weight"].T

        b = p + "mtp_block."
        positions = torch.arange(j + 1, j + 1 + seq)
        u = ops.rms_norm(x, params[b + "input_layernorm.weight"], cfg.norm_eps)
        q = heads(u @ params[b + "self_attn.q_proj.weight"].T, cfg.n_heads)
        q = ops.rope(q, positions, cfg.rope_base)
        k = heads(u @ params[b + "self_attn.k_proj.weight"].T, cfg.n_kv_heads)
        k = ops.rope(k, positions, cfg.rope_base)
        v = heads(u @ params[b + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        attended = ops.attention(q, k, v, causal)[0].transpose(0, 1).reshape(seq, cfg.d_model)
        x = x + attended @ params[b + "self_attn.o_proj.weight"].T
        u = ops.rms_norm(x, params[b + "post_attention_layernorm.weight"], cfg.norm_eps)
        x = x + ops.swiglu(
            u,
            params[b + "mlp.gate_proj.weight"],
            params[b + "mlp.up_proj.weight"],
            params[b + "mlp.down_proj.weight"],
        )

        h = ops.rms_norm(x, params[p + "post_norm.weight"], cfg.norm_eps)
        scores = h[-1] @ head.T
        token = torch.argmax(scores)
        drafted.append(token)
        logits.append(scores)
        ids = torch.cat((ids[1:], token[None]))
    return torch.stack(drafted), torch.stack(logits)
```

## Common mistakes

- **The concatenation order.** `eh_proj` has $2D$ input columns; which half multiplies
  the embedding is a property of the checkpoint. See "In the wild": the paper and the
  library's default disagree.
- **Embedding the current token.** `E[tokens]` next to `hidden` pairs position $i$ with
  token $i$, which the hidden state already contains, and gives the stage nothing new.
- **Every stage reading `hidden`.** That turns the chain into parallel heads with the
  wrong weights.
- **Not rolling the window,** so stage 1 embeds the same ids as stage 0.
- **Skipping the post norm,** or handing the next stage the block's output from before
  the post norm while the head gets the normed one.
- **`enorm` and `hnorm` swapped.** Both have shape `(D,)`.
- **No causal mask** (see above), or rotating the values.
- **Stage 0's weights for every stage.**
- **Assuming a tied head,** or the default `norm_eps`.
- **Rolling `tokens` in place.** The caller's prompt comes back shifted.
- **Calling `mtp_draft_ref`** from the kit: it is the oracle, and the harness rejects it.

## Complexity

Each stage is one block over $T$ rows, $O(T^2 D + T D\,d_\text{mlp})$, plus one head
product for the last row. In a decoding loop with a cache per stage, only the rows of
newly accepted tokens are processed, so a round costs a few block evaluations against
one full forward pass of the main model for every token it saves.

## In the wild

Checked on 2026-11-21 against the code installed with `transformers` 5.23.0, by reading
it and by running it (`fixtures/make.py`).

- **`MtpLayer`** (`modeling_layers.py`) computes
  `eh_proj(cat([enorm(inputs_embeds), hnorm(previous_hidden_state)]))`, runs
  `mtp_block`, and applies `post_norm`. **`MtpModel.forward`** loops over the stages:
  logits of the last position through the shared head, argmax, then
  `input_ids = cat([input_ids[:, 1:], next_mtp_token])` and every position id plus 1.
  The embedding, the head, and the rotary module are taken from the main model. The
  public tests compare with this module, built on a Llama decoder that holds the kit's
  weights.
- **The order of the concatenation.** The DeepSeek-V3 Technical Report (arXiv
  2412.19437), Eq. 21, writes the hidden state first:
  $h'^k_i = M_k[\operatorname{RMSNorm}(h^{k-1}_i);\ \operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))]$.
  `MtpLayer` puts the embedding first unless the configuration sets
  `mtp_hidden_states_first`. This problem follows the library's default; the fixture
  script asserts that the other order gives different logits. The report's Eq. 22 and
  23 apply the block and the shared head; the post norm is in the library's code.
- **What the module starts from.** The report says the embedding layer and the output
  head are shared with the main model. In the library, the hidden states handed to the
  module are the main model's last ones, which for a Llama decoder are the final-normed
  states (the kit's unit tests check that they are equal bit for bit).
- **Checkpoints.** The `config.json` of `Qwen/Qwen3.5-0.8B` has
  `mtp_num_hidden_layers: 1`; `zai-org/GLM-5.3` has `num_nextn_predict_layers: 1` and
  `deepseek-ai/DeepSeek-V4.1-Flash-1017` has 3. No `mtp.*` weights were loaded for this
  problem.
- **With a cache.** `MTPCandidateGenerator` (`generation/candidate_generator.py`) keeps
  an `MtpCache` per stage and passes only the rows of the tokens the main model just
  processed. With more than one stage, rows that were computed from rejected drafts
  are cropped from the cache and recomputed from the validated tokens.

Simplifications, all of them:

- The stage block is the kit's decoder block (RMSNorm pre-norm, grouped-query
  attention with rotary positions on adjacent pairs, SwiGLU). `MtpModel` uses whatever
  layer class the main model has: for Qwen3.5 that is not this block.
- One sequence, no batch axis, no key-value cache, so every call processes all $T$
  rows.
- Greedy drafting only. The library can also sample each draft and apply logits
  processors first; `draft_logits` here are the raw logits.
- A post norm per stage. Configurations with `chain_hidden_post_norm` have no per-stage
  post norm and optionally one shared norm; not modeled.
- The library's handling of embeddings with a built-in norm, of logit scaling
  (`logits_mup_width_multiplier`), and of a padded vocabulary is not modeled.
