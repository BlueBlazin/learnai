"""Regenerate the fixture of real-gdn-block.

    uv run --project python --extra models \
        python problems/real-hybrid-model/real-gdn-block/fixtures/make.py

Needs ``Qwen/Qwen3.5-0.8B`` on disk as the kit fetches it (``learnai assets fetch
qwen3.5-0.8b``: the weights, ``config.json``, ``model.safetensors.index.json``, which
``transformers`` needs to find the weight file, and ``tokenizer.json``, which gives the
prompt its ids) and the ``models`` extra. The reference implementation is
``transformers`` (``Qwen3_5ForCausalLM``, float32, CPU, one call, no cache); nothing
here comes from the kit's forward pass or from ``mlx-lm``.

``qwen3.5-0.8b.npz`` holds, for one 12-token prompt: the ids; the hidden states entering
layers 1 and 13 in full (they are inputs of the tests); and 64 fixed columns of the
hidden states leaving layers 1 and 13. Entry ``i + 1`` of ``output_hidden_states`` is
the output of layer ``i`` (only the last entry, 24, has the final norm applied; it is
not used here). Layers 1 and 13 are linear-attention layers: one early, one in the
middle of the stack.

One forward pass takes about 5 s: without the optional fast kernels ``transformers``
runs the delta rule in its reference PyTorch form.
"""

from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import Qwen3_5ForCausalLM

from learnai import assets
from learnai.kit.realmodel import QWEN35
from learnai.problem import find_root

HERE = Path(__file__).parent
PROMPT = "The whetstone sharpens the blade, and the blade"
LENGTH = 12
LAYERS = (1, 13)
COLUMNS = np.sort(np.random.default_rng(250).choice(1024, size=64, replace=False))


def main() -> None:
    directory = assets.resolve(QWEN35, assets.models_dir(find_root()))
    ids = Tokenizer.from_file(str(directory / "tokenizer.json")).encode(PROMPT).ids
    assert len(ids) == LENGTH, (len(ids), ids)
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        directory, dtype=torch.float32, output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"], info
    model.eval()
    kinds = model.config.layer_types
    assert all(kinds[layer] == "linear_attention" for layer in LAYERS)
    with torch.no_grad():
        states = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
    states = [state[0].numpy().astype(np.float32) for state in states]  # (12, 1024) each
    out: dict[str, np.ndarray] = {
        "revision": np.array(QWEN35.revision),
        "prompt": np.array(PROMPT),
        "ids": np.array(ids, dtype=np.int32),
        "columns": COLUMNS,
    }
    for layer in LAYERS:
        out[f"x_{layer}"] = states[layer]
        out[f"y_{layer}"] = states[layer + 1][:, COLUMNS]
        x, y = states[layer], states[layer + 1]
        print(f"layer {layer}: median |x| {np.median(np.abs(x)):.4f}, max |x| {np.abs(x).max():.3f}, "
              f"median |y - x| {np.median(np.abs(y - x)):.4f}")  # fmt: skip
    target = HERE / f"{QWEN35.name}.npz"
    np.savez_compressed(target, **out)
    print(f"wrote {target.name} ({target.stat().st_size} bytes); ids {ids}")


if __name__ == "__main__":
    main()
