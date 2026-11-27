## The idea

A good draft model must be cheap to run and closely match the main model's predictions. An MTP module achieves both by building directly on the main model's final hidden states. The main model's hidden state at position $i$ summarizes the prefix through position $i$ well enough that a single linear projection predicts token $i + 1$. Adding one transformer block on top of that state, informed by the actual token $i + 1$, lets the model predict token $i + 2$:

- Stage 0 pairs the hidden state of position $i$ with the embedding of token $i + 1$ and outputs a state that predicts token $i + 2$.
- Stage 1 pairs stage 0's output state with the embedding of token $i + 2$ and outputs a state that predicts token $i + 3$, and so on.

Each stage approximates what the main model would know one position later using a single block instead of the full layer stack. The token it needs is the one the previous stage just drafted. Compare this with `mtp-head-draft`, where every head reads the same hidden state without knowing what earlier heads guessed. Chaining conditions each draft on earlier drafts, at the cost of running stages sequentially.

Because the embedding matrix and output head are shared with the main model, each stage adds only two input norms, a $2D \to D$ projection, one decoder block, and a post-norm.

## The bookkeeping

Two tensors advance from stage to stage, and most bugs break one of them:

- **The token window (`ids`)** has length $T$ and stays one token ahead of the hidden states. It starts as `tokens[1:] + [next_token]` for stage 0. After each stage, it rolls left by one by dropping its first entry and appending the newly drafted token.
- **The hidden states (`h`)** start as `hidden` for stage 0 and then become the previous stage's full post-normed output. All $T$ rows matter because the block applies causal attention across rows, so row $T - 1$ of stage 1 attends to rows $0 \dots T - 1$ of stage 0's output. This is also why the causal mask is required even though only the last row is decoded. Row $T - 1$ of stage 0 sees every row with or without a mask, but earlier rows change without a mask, and stage 1 reads those earlier rows.

Only the last row's logits are computed, matching the reference library (which computes all rows only when given training labels for a loss).

## Positions

Stage $j$'s rows correspond to tokens $j + 1$ through $j + T$, and the library passes those exact positions by incrementing every position by 1 after each stage. Here each stage attends only within its own $T$ rows, and rotary attention scores depend only on relative position differences. Shifting all positions of a stage by a constant therefore leaves the output unchanged (passing `0..T-1` in every stage reproduces the logits to $3 \cdot 10^{-6}$). The absolute offset matters as soon as a stage maintains a key-value cache across decoding rounds, because cached keys were rotated with their absolute positions and new queries must match them.

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

- **Wrong concatenation order.** `eh_proj` has $2D$ input columns, and which half multiplies the embedding depends on the checkpoint. As noted in "In the wild", the DeepSeek-V3 paper and the Transformers default use opposite orders.
- **Embedding the current token.** Pairing `E[tokens]` with `hidden` matches position $i$ with token $i$, which is already encoded in `hidden`, giving the stage no new information.
- **Feeding `hidden` to every stage.** Reusing `hidden` instead of `h` turns the chain into parallel heads with mismatched weights.
- **Not rolling the token window.** Without rolling `ids`, stage 1 embeds the exact same tokens as stage 0.
- **Skipping the post-norm** or passing the pre-post-norm block output to the next stage while sending the post-normed state to the head.
- **Swapping `enorm` and `hnorm`.** Both have shape `(D,)`, so no shape error occurs.
- **Omitting the causal mask or rotating values.** Without a causal mask, earlier rows leak future information into stage 1.
- **Reusing stage 0's weights** across all stages.
- **Assuming a tied head or default `norm_eps`.** Always check `cfg.tie_embeddings` and pass `cfg.norm_eps`.
- **Rolling `tokens` in place.** Mutating the input tensor corrupts the caller's prompt.
- **Calling `mtp_draft_ref`** from `learnai.kit`, which is forbidden by the test harness.

## Complexity

Each stage runs one transformer block over $T$ rows in $O(T^2 D + T D\,d_\text{mlp})$ time, plus one $O(DV)$ head projection for the last row. In a cached decoding loop, each stage processes only the newly accepted tokens, so drafting costs a few single-block evaluations compared to one full main-model forward pass per saved token.

## In the wild

Checked on 2026-11-21 against `transformers` 5.23.0 by inspecting the source and running `fixtures/make.py`.

- **`MtpLayer` (`modeling_layers.py`)** computes `eh_proj(cat([enorm(inputs_embeds), hnorm(previous_hidden_state)]))`, runs `mtp_block`, and applies `post_norm`. **`MtpModel.forward`** loops over the stages, computes logits at the last position through the shared head, takes the argmax, updates `input_ids = cat([input_ids[:, 1:], next_mtp_token])`, and increments all position ids by 1. The embedding, output head, and rotary module come from the main model. Public tests compare directly against this module built on a Llama decoder with the kit's weights.
- **Concatenation order.** The DeepSeek-V3 Technical Report (arXiv 2412.19437, Eq. 21) puts the hidden state first: $h'^k_i = M_k[\operatorname{RMSNorm}(h^{k-1}_i);\ \operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))]$. `MtpLayer` puts the embedding first unless `mtp_hidden_states_first` is set in the config. This problem follows the library's default, and the fixture script verifies that swapping the order changes the logits. Equations 22 and 23 of the report apply the block and shared head, while `post_norm` appears in the library implementation.
- **Initial hidden states.** The report states that the embedding layer and output head are shared with the main model. In `transformers`, the module receives the main model's final hidden states, which for a Llama decoder are already final-normed (the kit's unit tests verify bit-for-bit equality).
- **Checkpoints.** `Qwen/Qwen3.5-0.8B` sets `mtp_num_hidden_layers: 1`, `zai-org/GLM-5.3` sets `num_nextn_predict_layers: 1`, and `deepseek-ai/DeepSeek-V4.1-Flash-1017` has 3 stages. No real `mtp.*` weights were loaded for this problem.
- **KV caching.** `MTPCandidateGenerator` (`generation/candidate_generator.py`) maintains an `MtpCache` per stage and feeds only the newly processed rows. When multiple stages are active, cached rows computed from rejected drafts are cropped and recomputed from the accepted tokens.

Simplifications relative to `transformers`:

- Every stage uses the kit's decoder block (RMSNorm pre-norm, grouped-query attention with rotary embeddings on adjacent pairs, and SwiGLU). `MtpModel` uses the main model's layer class, which differs for models like Qwen3.5.
- Single sequence with no batch axis and no KV cache, so every call processes all $T$ rows.
- Greedy drafting only. The library also supports sampling and logits processors, whereas `draft_logits` here are raw logits.
- One post-norm per stage. Configs with `chain_hidden_post_norm` omit per-stage post-norms and optionally use one shared norm.
- Built-in embedding norms, logit scaling (`logits_mup_width_multiplier`), and padded vocabularies are not modeled.
