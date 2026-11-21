## The idea

A configuration file is a compressed program. The reference implementation expands it
once, when the model object is built, into one decision per layer, and from then on
only the expansion exists. Porting a model means reproducing that expansion, and it is
short enough to read: `__post_init__` of `Qwen3_5TextConfig` and `GlmMoeDsaConfig`,
and `__post_init__` plus the property `kv_sharing_modes` of `Gemma4TextConfig`, in
Hugging Face Transformers 5.23.

Three things recur across the families.

**A list beats a rule, and the rule is the fallback.** Each family has a per-layer
list (`layer_types`, `mlp_layer_types`, `indexer_types`) and a few numbers the list
can be generated from. The classes fill the list in only when it is `None`. The three
published files carry both, and they agree, so an implementation that reads only the
numbers passes on every real file and fails on the first one whose list was edited by
hand. GLM has three levels: the list, then `index_topk_pattern` (a string such as
`"FSSF"`), then frequency and offset.

**The rules count differently.** Qwen3.5 counts layers from 1: layer $i$ attends when
$(i + 1) \bmod n = 0$, so with $n = 4$ the attention layers are 3, 7, 11, .... GLM's
indexer rule is $\max(i - o + 1, 0) \bmod q = 0$. The clamp is the point: every layer
up to $o - 1$ maps to 0 and runs its own indexer, and from there on every $q$-th layer
does. With $o = 3$, $q = 4$: full at 0, 1, 2, then 6, 10, 14, .... Without the clamp
layer 0 gives $-2 \bmod 4 = 2$ in Python and becomes "shared", with no earlier layer
to share with. A frequency of 0 is clamped to 1 (every layer full) instead of dividing
by zero.

**Defaults belong to the family, not to the concept.** A missing
`partial_rotary_factor` means 0.25 in Qwen3.5 (its class sets the default itself,
under the comment `# assign default for BC`) and 1.0 in the generic proportional RoPE
that Gemma 4 uses. A
missing `index_skip_topk_offset` is 2, while the published GLM-5.3 says 3. A missing
`global_head_dim` is 512.

Gemma 4's key-value sharing is the one rule that is not a schedule. The last $s$
layers compute no keys and values; each attends over the ones stored by an earlier
layer of its own type. So there is one producer per *type*: the last sliding layer and
the last full-attention layer before the shared block. In the E2B model those are
layers 13 and 14. They cannot be merged, because the two types differ in head width
(256 and 512) and in RoPE (default with base $10^4$, proportional with base $10^6$).

Head width and rotating channels also come from three places: Qwen3.5 has one
`head_dim` and rotates `int(head_dim * f)` channels of it; GLM builds a head from a
part without positions and a rotating part, and its file's `head_dim` key holds
neither their sum nor, in general, the rotating part (192, 192 + 64, and 64 in
GLM-5.3; the class overwrites `head_dim` with the rotating width when it loads);
Gemma 4 keeps a second head width for full-attention layers.

One input class is left out of the tests on purpose. When `int(head_dim * f)` is odd
in a Qwen3.5 file, the library still builds one frequency per started pair
(`arange(0, dim, 2)`), so it rotates one channel more than the formula says: a head of
100 with the default factor gives `int(25.0) = 25` and a table of 13 frequencies, 26
channels. Every test keeps the value even, as every published file does.

## Reference solution (Python)

```python
def layer_plan(config):
    text = config.get("text_config", config)
    family = text.get("model_type")
    if family == "qwen3_5_text":
        return _qwen(text)
    if family == "glm_moe_dsa":
        return _glm(text)
    if family == "gemma4_text":
        return _gemma(text)
    raise ValueError(f"unsupported model_type {family!r}")


def _qwen(text):
    n = text["num_hidden_layers"]
    mixers = text.get("layer_types")
    if mixers is None:
        interval = text.get("full_attention_interval", 4)
        mixers = [FULL if (i + 1) % interval == 0 else LINEAR for i in range(n)]
    params = text["rope_parameters"]
    factor = params.get("partial_rotary_factor", 0.25)
    head = text["head_dim"]
    rope = _rope(params["rope_type"], params["rope_theta"], head, int(head * factor))
    return [_entry(mixer, rope=dict(rope) if mixer == FULL else None) for mixer in mixers]


def _glm(text):
    n = text["num_hidden_layers"]
    mlps = text.get("mlp_layer_types")
    if mlps is None:
        dense = text.get("first_k_dense_replace", 3)
        mlps = ["dense" if i < dense else "sparse" for i in range(n)]
    indexers = text.get("indexer_types")
    if indexers is None:
        pattern = text.get("index_topk_pattern")
        if pattern is not None:
            letters = {"F": "full", "S": "shared"}
            indexers = [letters[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
        else:
            freq = max(text.get("index_topk_freq", 1), 1)
            offset = text.get("index_skip_topk_offset", 2)
            indexers = ["full" if max(i - offset + 1, 0) % freq == 0 else "shared" for i in range(n)]
    params = text["rope_parameters"]
    rotary = text["qk_rope_head_dim"]
    rope = _rope(params["rope_type"], params["rope_theta"], text["qk_nope_head_dim"] + rotary, rotary)
    return [_entry("indexed_attention", mlps[i], indexers[i], dict(rope)) for i in range(n)]


def _gemma(text):
    kinds = text["layer_types"]
    n = len(kinds)
    shared = text.get("num_kv_shared_layers", 0)
    kv = [None] * n
    if shared:
        first = n - shared
        for i in range(first, n):
            kv[i] = "reuse"
        for kind in set(kinds):
            kv[max(i for i in range(first) if kinds[i] == kind)] = "store"
    plan = []
    for kind, mode in zip(kinds, kv, strict=True):
        params = text["rope_parameters"][kind]
        head = text.get("global_head_dim", 512) if kind == FULL else text["head_dim"]
        if params["rope_type"] == "proportional":
            rotary = 2 * int(params.get("partial_rotary_factor", 1.0) * head // 2)
        else:
            rotary = head
        plan.append(_entry(kind, rope=_rope(params["rope_type"], params["rope_theta"], head, rotary), kv=mode))
    return plan
```

`_entry` and `_rope` build the two dicts. Linear-attention layers of Qwen3.5 get
`rope: None` because `Qwen3_5DecoderLayer` passes the position embeddings to its
attention sublayer only; a Gated DeltaNet layer learns order from its recurrence.

## Common mistakes

- **Counting from 0** (`i % interval == 0`): attention at 0, 4, 8, ... instead of 3,
  7, 11, .... Loud with real weights (the loader looks for `self_attn` tensors where
  the file has `linear_attn` ones), silent in anything built from the config alone,
  such as a cache or memory plan.
- **Ignoring the explicit list**, or letting the pattern override it. Right on every
  published file.
- **Ignoring `index_topk_pattern`.** GLM-5.3 ships it as `null`, so the branch looks
  dead until a checkpoint sets it.
- **Dropping the clamp**, `(i - offset + 1) % freq`: layers before `offset - 1` are no
  longer all full. **`i - offset`** instead of `i - offset + 1` shifts every later
  full layer by one.
- **Offset default 0.** The class's default is 2.
- **Rotary factor default 1.0 for Qwen3.5.** The generic default is "rotate
  everything"; this family's is a quarter.
- **GLM's head width from `head_dim`.** The file's value is the width of the part
  without positions.
- **One `store` layer.** The layer right before the shared block stores, and the last
  layer of the *other* type does too.
- **`store` with nothing shared.** `kv_sharing_modes` returns all `None` when
  `num_kv_shared_layers` is 0.
- **One RoPE entry or one head width for every Gemma 4 layer**: the sliding layers'
  parameters for full-attention layers, or `head_dim` where `global_head_dim` applies.
- **`int(f * head_dim)` for the proportional type.** The library counts rotating
  *pairs*, `int(f * head_dim // 2)`, so an odd product rounds down to an even number
  of channels: 4 of 20 at $f = 0.25$, not 5.
- **RoPE on Qwen3.5's linear-attention layers.**
- **Importing the configuration classes.** That is the lookup this problem asks you to
  be able to do without.

## Complexity

$O(L)$ time and memory; $O(L \cdot \text{types})$ for the scan that places `store`,
with two types.

## In the wild

The fixture of this problem was made by building `Qwen3_5TextConfig`,
`GlmMoeDsaConfig`, and `Gemma4TextConfig` from each test configuration and reading the
resolved lists, `kv_sharing_modes`, the per-layer head widths, and the frequency tables
of the three rotary-embedding modules (Transformers 5.23.0). Two facts from that code
that the plan does not carry: a Gemma 4 layer in `reuse` mode also gets an MLP twice
as wide when `use_double_wide_mlp` is set, and GLM's shared layers receive the top-k
indices of the nearest preceding full layer, passed up the stack by the model's
forward loop. The cross-layer reuse of indices is described in "IndexCache:
Accelerating Sparse Attention via Cross-Layer Index Reuse" (arXiv 2603.12201), of
which only the abstract was read for this curriculum. The next two problems run the
Qwen3.5 plan on real weights: `real-gdn-block` for one linear-attention layer and
`real-hybrid-logits` for all 24.
