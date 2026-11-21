"""Regenerate the fixture of mtp-chain-draft.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/drafts-and-diffusion/mtp-chain-draft/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. The expected
drafts are computed by ``MtpModel.forward`` (``MtpLayer`` per stage) of
``transformers/modeling_layers.py``, so the tests compare a solution with the shipped
code and not only with the kit's oracle.

``MtpModel`` builds its stages from the decoder layer class of the main model it is
given. Here the main model is a ``LlamaForCausalLM`` that holds the kit decoder's seeded
weights (``mtp_params(cfg, stages, seed)``), and the ``mtp.*`` entries are loaded into
the module. A Llama block is the kit block except for the layout of the rotary pairs
(Llama pairs channel ``i`` with ``i + head_dim / 2``, the kit pairs ``2i`` with
``2i + 1``), so the rows of ``q_proj`` and ``k_proj`` are reordered inside each head on
the way in, which changes no attention score.

Per run ``r{n}`` (default ``DecoderConfig`` with 2 layers; ``seed``, ``stages`` stored):

- ``tokens``: the prompt; ``next_token``: the argmax of the Llama model's last logits;
- ``hidden``: the Llama model's ``last_hidden_state`` (after its final norm), which is
  what ``generate`` hands the module;
- ``draft_tokens``, ``draft_logits``: the module's output when called as the generator
  calls it, with the ids and positions shifted by one, without a cache.

The script asserts that the module's drafts with an ``MtpCache`` are the same, and that
setting ``mtp_hidden_states_first`` (the order of the DeepSeek-V3 report) changes them.
"""

from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import MtpCache
from transformers.modeling_layers import MtpModel

from learnai.kit import DecoderConfig, mtp_params

HERE = Path(__file__).parent
RUNS = (  # (seed, prompt length, stages)
    (8, 8, 3),
    (5, 8, 2),
    (4, 8, 3),
    (0, 1, 3),
)


def build(cfg: DecoderConfig, params: dict[str, np.ndarray], stages: int):
    head_dim = cfg.head_dim
    within = np.concatenate((np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)))

    def convert(name: str, value: np.ndarray) -> torch.Tensor:
        if name.endswith(("self_attn.q_proj.weight", "self_attn.k_proj.weight")):
            heads = value.shape[0] // head_dim
            value = value[np.concatenate([h * head_dim + within for h in range(heads)])]
        return torch.from_numpy(value.copy())

    config = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.d_mlp,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        rms_norm_eps=cfg.norm_eps,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_base},
    )
    config.num_mtp_layers = stages
    main = LlamaForCausalLM(config).eval()
    state = {n: convert(n, v) for n, v in params.items() if not n.startswith("mtp.")}
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    main.load_state_dict(state)
    mtp = MtpModel(main, stages).eval()
    own = {n[len("mtp.") :]: convert(n, v) for n, v in params.items() if n.startswith("mtp.")}
    result = mtp.load_state_dict(own, strict=False)
    assert set(result.missing_keys) == {"embed_tokens.weight", "shared_head.weight"}
    assert not result.unexpected_keys and mtp.use_post_norm
    return main, mtp, config


def main() -> None:
    out = {"transformers_version": np.array(transformers.__version__), "runs": np.array(len(RUNS))}
    cfg = DecoderConfig(n_layers=2)
    for n, (seed, length, stages) in enumerate(RUNS):
        params = mtp_params(cfg, stages, seed)
        model, mtp, config = build(cfg, params, stages)
        tokens = np.random.default_rng([seed, length]).integers(0, cfg.vocab_size, length)
        ids = torch.from_numpy(tokens)[None]
        with torch.no_grad():
            logits = model(ids).logits[0]
            hidden = model.model(ids).last_hidden_state
        next_token = int(logits[-1].argmax())
        shifted = torch.cat((ids[:, 1:], torch.tensor([[next_token]])), dim=1)
        positions = torch.arange(1, length + 1)[None]
        with torch.no_grad():
            drafted, draft_logits, _ = mtp(shifted, hidden, None, positions, None)
            cached, _, _ = mtp(
                shifted, hidden, None, positions, MtpCache(config=config.get_mtp_config())
            )
            mtp.config.mtp_hidden_states_first = True
            _, paper_order, _ = mtp(shifted, hidden, None, positions, None)
            mtp.config.mtp_hidden_states_first = False
        assert torch.equal(drafted, cached)
        assert (paper_order - draft_logits).abs().max() > 0.1
        top2 = torch.topk(draft_logits[0], 2, dim=-1).values
        gap = float((top2[:, 0] - top2[:, 1]).min())
        assert gap > 0.05, f"run {n}: the two largest draft logits are {gap:.3f} apart"
        assert next_token != int(tokens[-1]), f"run {n}: next_token repeats the last token"
        out[f"r{n}_seed"], out[f"r{n}_stages"] = np.array(seed), np.array(stages)
        out[f"r{n}_tokens"], out[f"r{n}_next_token"] = tokens, np.array(next_token)
        out[f"r{n}_hidden"] = hidden[0].numpy()
        out[f"r{n}_draft_tokens"] = drafted[0].numpy()
        out[f"r{n}_draft_logits"] = draft_logits[0].numpy()
        print(f"run {n}: T={length}, stages={stages}, draft {drafted[0].tolist()}, gap {gap:.3f}")

    np.savez(HERE / "shipped.npz", **out)
    print(f"wrote shipped.npz (transformers {transformers.__version__})")


if __name__ == "__main__":
    main()
