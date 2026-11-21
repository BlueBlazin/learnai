Expand the `config.json` of a decoder into its per-layer plan: which mixer each layer
runs, which MLP, whether it scores the past itself, how its heads rotate, and whether it
makes or borrows keys and values. Three model families write this down in three ways,
and each fills what the file leaves out with defaults of its own.

## Specification

```python
def layer_plan(config): ...  # -> list of dicts, one per layer
```

`config` is the parsed file, a dict. The decoder's settings are `config["text_config"]`
when `config` has that key (a multimodal file) and `config` itself otherwise; call that
dict `c`, and let $L$ = `c["num_hidden_layers"]`. The family is `c["model_type"]`. The
result has $L$ entries, layer 0 first, each a dict with exactly these keys:

| key | value |
|---|---|
| `"mixer"` | a string: the kind of token mixer |
| `"mlp"` | `"dense"` or `"sparse"` (a mixture of experts) |
| `"indexer"` | `"full"` (the layer runs its own indexer), `"shared"` (it uses the indices of the nearest full layer before it), or `None` |
| `"rope"` | `None`, or `{"type": ..., "theta": ..., "head_dim": ..., "rotary_dim": ...}`: the RoPE type, its base, the width of a query head, and how many channels of that head rotate |
| `"kv"` | `"store"` (the layer's keys and values are kept for later layers), `"reuse"` (it attends over stored ones and makes none), or `None` |

Below, "missing" means the key is absent from `c`. For the four keys that hold a
per-layer list or a pattern (`layer_types`, `mlp_layer_types`, `indexer_types`,
`index_topk_pattern`), a value of `None` (JSON `null`) counts as missing too. An
explicit list has $L$ entries and always wins over the rule that would replace it.

**`"qwen3_5_text"`** (Qwen3.5). `mlp` is `"dense"`, `indexer` and `kv` are `None`.

- `mixer`: `c["layer_types"][i]`. If missing: `"full_attention"` when
  $(i + 1) \bmod n = 0$, else `"linear_attention"`, with $n$ =
  `c["full_attention_interval"]`, or 4 if that is missing.
- `rope`: `None` for a `"linear_attention"` layer. For a `"full_attention"` layer, with
  `r = c["rope_parameters"]`: `type` is `r["rope_type"]`, `theta` is `r["rope_theta"]`,
  `head_dim` is `c["head_dim"]`, and `rotary_dim` is `int(head_dim * f)` with $f$ =
  `r["partial_rotary_factor"]`, or **0.25** if that is missing.

**`"glm_moe_dsa"`** (GLM with sparse attention). `mixer` is `"indexed_attention"`, `kv`
is `None`.

- `mlp`: `c["mlp_layer_types"][i]`. If missing: `"dense"` for the first $k$ layers and
  `"sparse"` after them, with $k$ = `c["first_k_dense_replace"]`, or 3 if missing.
- `indexer`: `c["indexer_types"][i]`. If missing, `c["index_topk_pattern"]`: a string
  of $L$ letters, `F` for `"full"` and `S` for `"shared"`, or a list of $L$ such
  words. If that is missing too:

$$
\text{indexer}_i = \texttt{"full"} \iff \max(i - o + 1,\ 0) \bmod \max(q, 1) = 0,
$$

  with $q$ = `c["index_topk_freq"]` (1 if missing) and $o$ =
  `c["index_skip_topk_offset"]` (2 if missing).
- `rope`, the same for every layer: `type` and `theta` from `c["rope_parameters"]` as
  above, `head_dim` is `c["qk_nope_head_dim"] + c["qk_rope_head_dim"]`, and
  `rotary_dim` is `c["qk_rope_head_dim"]`. The file's own `head_dim` key is not used.

**`"gemma4_text"`** (Gemma 4). `mixer` is `c["layer_types"][i]`, `"sliding_attention"`
or `"full_attention"` (the list is always present). `mlp` is `"dense"`, `indexer` is
`None`.

- `rope`, with `r = c["rope_parameters"][mixer]` (one entry per layer type): `type` is
  `r["rope_type"]`, `theta` is `r["rope_theta"]`. `head_dim` is `c["global_head_dim"]`
  for a full-attention layer (**512** if missing) and `c["head_dim"]` for a sliding
  one. `rotary_dim` is `head_dim` when the type is `"default"`, and
  `2 * int(f * head_dim // 2)` when it is `"proportional"`, with $f$ =
  `r["partial_rotary_factor"]`, or 1.0 if missing.
- `kv`, with $s$ = `c["num_kv_shared_layers"]` (0 if missing). If $s = 0$, every `kv`
  is `None`. Otherwise layers $L - s, \dots, L - 1$ are `"reuse"`; for each layer type
  that occurs in the model, the **last layer of that type before layer $L - s$** is
  `"store"`; every other layer is `None`.

Any other `model_type`, or none, raises `ValueError`. Expand the configuration
yourself: importing `transformers` is not allowed.

![One cell per layer, layer 0 on the left. A filled cell is the less common kind of layer in its model; the plan also records what the brackets mark.](assets/layer-plans.svg)

## Example

The first public case is the published file of `Qwen/Qwen3.5-0.8B`, reduced:

```python
config = {
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 24,
        "head_dim": 256,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                        "full_attention", ...],          # 24 entries, the same period
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25,
                            "mrope_interleaved": True, "mrope_section": [11, 11, 10]},
        ...                                              # sizes the plan does not read
    },
}
plan = layer_plan(config)
len(plan)  # 24
plan[0]    # {"mixer": "linear_attention", "mlp": "dense", "indexer": None,
           #  "rope": None, "kv": None}
plan[3]    # {"mixer": "full_attention", "mlp": "dense", "indexer": None,
           #  "rope": {"type": "default", "theta": 10000000, "head_dim": 256,
           #           "rotary_dim": 64},
           #  "kv": None}
```

Layers 3, 7, 11, 15, 19, and 23 look like `plan[3]`, the other 18 like `plan[0]`;
`rotary_dim` is `int(256 * 0.25) = 64`. Without `layer_types` the rule gives the same
list: $(3 + 1) \bmod 4 = 0$.

The other two families, on their published files (Figure 1):

- `zai-org/GLM-5.3`, 78 layers, `index_topk_freq` 4, `index_skip_topk_offset` 3. Layers
  0, 1, 2 give $\max(i - 2, 0) = 0$, so they are full; layer 3 gives 1, shared; layer
  6 gives 4, full; then every fourth layer. `first_k_dense_replace` is 3, so layers 0
  to 2 are dense. The file says `"head_dim": 192`, `"qk_nope_head_dim": 192`,
  `"qk_rope_head_dim": 64`: the plan's `head_dim` is 256 and its `rotary_dim` 64.
- `google/gemma-4-E2B-it`, 35 layers, full attention at 4, 9, ..., 34,
  `num_kv_shared_layers` 20. Layers 15 to 34 reuse. The last full-attention layer
  before 15 is 14 and the last sliding one is 13: both store. A full layer has
  `head_dim` 512 (`global_head_dim`), type `"proportional"`, and `rotary_dim`
  `2 * int(0.25 * 512 // 2) = 128`; a sliding layer has `head_dim` 256, type
  `"default"`, and `rotary_dim` 256.

## Constraints

- The result is compared exactly with the plan that the configuration classes of
  Hugging Face Transformers 5.23 resolve for the same file (`Qwen3_5TextConfig`,
  `GlmMoeDsaConfig`, `Gemma4TextConfig`, and the models' rotary embeddings). `theta`
  may be an int or a float, as the file has it.
- $1 \le L \le 78$. Every key the rules read without a stated default is present.
  Optional keys that are not lists or patterns are either missing or valid numbers.
- In a Qwen3.5 file `int(head_dim * f)` is even (RoPE rotates pairs).
- `full_attention_interval` $\ge 1$; `index_topk_freq`, `index_skip_topk_offset`, and
  `first_k_dense_replace` are $\ge 0$ and may exceed $L$.
- In a Gemma 4 file the last layer is `"full_attention"`, $0 \le s < L$, every layer
  type of the model also occurs before layer $L - s$, and a `"default"` RoPE entry has
  no `partial_rotary_factor`.
- The three published files are reduced copies: the vision and audio towers, token
  ids, dtypes, dropout and initializer settings, and GLM's `quantization_config` are
  removed; every key a plan reads and the main sizes keep their published values.
  The other tests are variations of them and short synthetic files.

## Why this matters

A runtime does not execute `config.json`; it executes the plan the reference
implementation derives from it, and the derivation lives in a few lines of each
model's configuration class. A port that counts layers from 0 looks for attention
weights in Qwen3.5 layers that hold recurrent ones. A port that lets GLM's layer 0 be "shared" has no index
to share. A port that stores keys and values for the last unshared Gemma 4 layer only
leaves the shared layers of the other type, whose heads have another width, with
nothing to read. All three published files list their schedules explicitly, so a wrong
rule goes unnoticed until a file arrives that leaves the list out.
