"""Regenerate the reference fixture for the real hybrid model's kit tests.

    uv run --project python --extra models python python/tests/fixtures/realhybrid/make.py

Needs ``Qwen/Qwen3.5-0.8B`` on disk (``learnai assets fetch qwen3.5-0.8b``) and the
``models`` extra. The reference implementation is ``transformers`` (float32, CPU):
``Qwen3_5ForCausalLM``, the text-only class, which loads the ``model.language_model.``
tensors of the vision-language checkpoint and ignores the rest. The prompts are token
ids (``prompts.json``; they were made once with ``tokenizers`` from the texts recorded
there, and the last is random ids), so nothing here depends on a tokenizer.

Stored per prompt, small on purpose: the ids, the argmax at every position, the logits
of every position at 64 fixed vocabulary columns, the top 32 logits of the last
position, the smallest gap between the two largest logits of any position, and 16
columns of the hidden state after the embedding and after layers 0, 1, 2, 3, 12, 13,
and 23 (``hidden_states`` entries 0, 1, 2, 3, 4, 13, 14, 24). Layers 0 to 2, 12, and 13
are linear-attention layers; 3 and 23 are attention layers. ``transformers`` applies the
final norm to the last entry (24) and to no other.

One forward pass of 16 tokens takes about 5 s here: without the optional
``flash-linear-attention`` and ``causal_conv1d`` packages ``transformers`` runs the
delta rule in its reference PyTorch form.

This is also the pattern for the ``fixtures/make.py`` of a problem on this model.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM

from learnai import assets
from learnai.kit.realmodel import QWEN35
from learnai.problem import find_root

HERE = Path(__file__).parent
#: The vocabulary columns whose logits are stored for every position.
COLUMNS = np.random.default_rng(35).choice(248320, size=64, replace=False)
#: Entries of ``output_hidden_states`` that are stored (entry i + 1 is layer i's output).
HIDDEN = (0, 1, 2, 3, 4, 13, 14, 24)
HIDDEN_COLUMNS = np.random.default_rng(8).choice(1024, size=16, replace=False)


def main() -> None:
    directory = assets.resolve(QWEN35, assets.models_dir(find_root()))
    prompts = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        directory, dtype=torch.float32, output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"], info
    model.eval()
    out: dict[str, np.ndarray] = {
        "columns": COLUMNS,
        "hidden_columns": HIDDEN_COLUMNS,
        "revision": np.array(QWEN35.revision),
    }
    for index, prompt in enumerate(prompts):
        ids = torch.tensor([prompt["ids"]])
        with torch.no_grad():
            result = model(ids, output_hidden_states=True)
        logits = result.logits[0].numpy()
        top = np.argsort(-logits[-1])[:32]
        ordered = np.sort(logits, axis=-1)
        out[f"ids_{index}"] = ids[0].numpy().astype(np.int32)
        out[f"argmax_{index}"] = logits.argmax(-1).astype(np.int32)
        out[f"logits_{index}"] = logits[:, COLUMNS]
        out[f"top_ids_{index}"] = top.astype(np.int32)
        out[f"top_logits_{index}"] = logits[-1, top]
        out[f"margin_{index}"] = np.float32((ordered[:, -1] - ordered[:, -2]).min())
        for entry in HIDDEN:
            hidden = result.hidden_states[entry][0].numpy()
            out[f"hidden_{index}_{entry}"] = hidden[:, HIDDEN_COLUMNS]
    target = HERE / f"{QWEN35.name}.npz"
    np.savez_compressed(target, **out)
    print(f"wrote {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
