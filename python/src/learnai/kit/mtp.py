"""A chained multi-token-prediction (MTP) module: seeded parameters and the oracle.

Checkpoints such as Qwen3.5, GLM-5.3, and DeepSeek-V3 ship a small draft model inside
the main one, under ``mtp.*``. After the main model has read ``T`` tokens and produced
the next one, stage 0 of the module drafts the token after that, stage 1 the one after,
and so on, each stage one decoder block deep. The stages are *chained*: a stage reads
the previous stage's hidden states, not the main model's.

    cfg = DecoderConfig(n_layers=2)
    params = mtp_params(cfg, n_layers=3, seed=0)        # decoder_params + mtp.layers.{0,1,2}.*
    hidden = decoder_hidden_ref(cfg, params, tokens)    # (T, d_model), after the final norm
    next_token = int(decoder_forward_ref(cfg, params, tokens)[-1].argmax())
    draft, logits = mtp_draft_ref(cfg, params, tokens, next_token, hidden, n_draft=3)

Stage ``j`` (from 0), with ``ids`` the token ids shifted left by ``j + 1`` and ending
with the latest token (``next_token`` for stage 0, then each drafted token), and
``h`` the previous stage's output (the main model's hidden states for stage 0):

    x = eh_proj(concat(rms_norm(E[ids], enorm), rms_norm(h, hnorm)))    embedding first
    x = block_j(x) at positions j + 1 .. j + T, causal                   one decoder block
    h = rms_norm(x, post_norm)
    token = argmax(h[-1] @ head^T)                                       the shared head

``E`` and the head are the main model's (with tied embeddings, the same matrix).
The block is one kit decoder block (``decoder_block_ref``: pre-norm, RoPE on adjacent
pairs, grouped-query attention, SwiGLU). Parameter names, for stage ``k``:

    mtp.layers.{k}.enorm.weight        (d_model,)
    mtp.layers.{k}.hnorm.weight        (d_model,)
    mtp.layers.{k}.eh_proj.weight      (d_model, 2 * d_model)
    mtp.layers.{k}.mtp_block.*         the names of one decoder layer (input_layernorm.weight,
                                       self_attn.q_proj.weight, ..., mlp.down_proj.weight)
    mtp.layers.{k}.post_norm.weight    (d_model,)

This is ``MtpLayer`` and the drafting loop of ``MtpModel.forward`` in ``transformers``
5.23, called without a cache. The DeepSeek-V3 report writes the hidden state first in
the concatenation; ``transformers`` puts the embedding first unless the configuration
says otherwise, and this module follows ``transformers``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from learnai.kit import ops
from learnai.kit.decoder import DecoderConfig, decoder_block_ref, decoder_params


def mtp_params(cfg: DecoderConfig, n_layers: int = 1, seed: int = 0) -> dict[str, np.ndarray]:
    """``decoder_params(cfg, seed)`` plus ``n_layers`` MTP stages under ``mtp.layers.{k}.``.

    The main model's entries are exactly those of ``decoder_params(cfg, seed)``. Gains
    are drawn from ``[0.5, 1.5]``; draft logits have a standard deviation of about 2,
    like the main model's.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    params = decoder_params(cfg, seed)
    rng = np.random.default_rng([seed, 0x6D7470])
    d, kv = cfg.d_model, cfg.n_kv_heads * cfg.head_dim

    def linear(out_dim: int, in_dim: int) -> np.ndarray:
        return (rng.standard_normal((out_dim, in_dim)) * in_dim**-0.5).astype(np.float32)

    def gain() -> np.ndarray:
        return rng.uniform(0.5, 1.5, size=d).astype(np.float32)

    for k in range(n_layers):
        stage = f"mtp.layers.{k}."
        params[stage + "enorm.weight"] = gain()
        params[stage + "hnorm.weight"] = gain()
        params[stage + "eh_proj.weight"] = linear(d, 2 * d)
        block = stage + "mtp_block."
        params[block + "input_layernorm.weight"] = gain()
        params[block + "self_attn.q_proj.weight"] = linear(d, d)
        params[block + "self_attn.k_proj.weight"] = linear(kv, d)
        params[block + "self_attn.v_proj.weight"] = linear(kv, d)
        params[block + "self_attn.o_proj.weight"] = linear(d, d)
        params[block + "post_attention_layernorm.weight"] = gain()
        params[block + "mlp.gate_proj.weight"] = linear(cfg.d_mlp, d)
        params[block + "mlp.up_proj.weight"] = linear(cfg.d_mlp, d)
        params[block + "mlp.down_proj.weight"] = linear(d, cfg.d_mlp)
        params[stage + "post_norm.weight"] = gain()
    return params


def mtp_layer_params(params: dict[str, Any], stage: int) -> dict[str, Any]:
    """The parameters of one MTP stage, with the ``mtp.layers.{stage}.`` prefix removed.

        layer = mtp_layer_params(params, 0)
        layer["eh_proj.weight"], layer["mtp_block.self_attn.q_proj.weight"]

    Works on numpy arrays and on tensors alike. ``KeyError`` if the stage does not exist.
    """
    prefix = f"mtp.layers.{stage}."
    found = {
        name[len(prefix) :]: value for name, value in params.items() if name.startswith(prefix)
    }
    if not found:
        raise KeyError(f"no MTP parameters for stage {stage}")
    return found


def mtp_draft_ref(
    cfg: DecoderConfig,
    params: dict[str, np.ndarray],
    tokens: Any,
    next_token: int,
    hidden: Any,
    n_draft: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The oracle draft: ``(draft_tokens, draft_logits)``.

    ``tokens`` are the ``T >= 1`` ids the main model has read, ``next_token`` the id it
    then produced, and ``hidden`` its hidden states after the final norm, ``(T,
    d_model)`` (``decoder_hidden_ref``). ``n_draft`` stages run, one per MTP layer in
    ``params``. Returns the drafted ids, int64 ``(n_draft,)``, and the logits each was
    the argmax of, float32 ``(n_draft, vocab_size)``. ``draft_tokens[0]`` is a guess at
    the token after ``next_token``.
    """
    ids = np.asarray(tokens)
    if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu":
        raise ValueError("tokens must be integers of shape (T,) with T >= 1")
    h = np.asarray(hidden, dtype=np.float32)
    if h.shape != (ids.size, cfg.d_model):
        raise ValueError(f"hidden must have shape ({ids.size}, {cfg.d_model}), got {h.shape}")
    seq = ids.size
    embedding = params["model.embed_tokens.weight"]
    head = embedding if cfg.tie_embeddings else params["lm_head.weight"]
    causal = np.tril(np.ones((seq, seq), dtype=bool))
    # The module's inputs run one token ahead of the main model's: tokens[1:] + [next].
    window = np.concatenate((ids[1:], [next_token])).astype(np.int64)
    drafted, logits = [], []
    for j in range(n_draft):
        layer = mtp_layer_params(params, j)
        joined = np.concatenate(
            (
                ops.rms_norm(embedding[window], layer["enorm.weight"], cfg.norm_eps),
                ops.rms_norm(h, layer["hnorm.weight"], cfg.norm_eps),
            ),
            axis=-1,
        )
        x = joined @ layer["eh_proj.weight"].T
        block = {
            name[len("mtp_block.") :]: value
            for name, value in layer.items()
            if name.startswith("mtp_block.")
        }
        # Stage j's inputs sit j + 1 positions after the main model's.
        positions = np.arange(j + 1, j + 1 + seq)
        x = decoder_block_ref(cfg, block, x[None], causal, positions)[0]
        h = ops.rms_norm(x, layer["post_norm.weight"], cfg.norm_eps)
        scores = (h[-1] @ head.T).astype(np.float32)
        token = int(scores.argmax())
        drafted.append(token)
        logits.append(scores)
        window = np.concatenate((window[1:], [token]))  # roll: the next stage sees the draft
    return np.asarray(drafted, dtype=np.int64), np.stack(logits)
