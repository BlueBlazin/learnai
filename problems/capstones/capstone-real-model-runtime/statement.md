Build a complete inference runtime for IBM's Granite 4.0 350M from three files on disk (`config.json`, `tokenizer.json`, and `model.safetensors`) to a streamed chat reply. You will implement five stages in a single file with no external model or tokenizer libraries, where each stage feeds directly into the next.

## Overview

```python
def load_weights(file, names=None): ...  # stage 1 -> {name: float32 tensor}


class Tokenizer:  # stage 2
    def __init__(self, tokenizer_json): ...
    def encode(self, text): ...  # -> list[int]
    def decode(self, ids): ...  # -> str
    def token_bytes(self, token): ...  # -> bytes


def chat_prompt(messages): ...  # stage 2 -> str
def forward(weights, config, ids): ...  # stage 3 -> logits of shape (T, V)
def prefill(weights, config, prompt_ids): ...  # stage 4 -> (logits (V,), cache)
def decode_step(weights, config, token, cache): ...  # stage 4 -> (logits (V,), cache)
def sample_token(logits, sampler, history, u): ...  # stage 4 -> int
def generate(weights, config, prompt_ids, max_new_tokens, sampler, uniforms): ...  # -> list[int]
def stream_chat(
    weights, config, tokenizer, messages, max_new_tokens, sampler, uniforms, stop
): ...  # stage 5 -> generator of str
```

Here `T` is the input sequence length and `V` is the vocabulary size.

![Three files go in and streamed text comes out. Each box is labeled with its stage.](assets/architecture.svg)

**Two models.** Every stage is tested on two model directories that each contain `config.json`, `tokenizer.json`, and `model.safetensors`:

- A **tiny model** bundled with the problem (`fixtures/tiny/`). It has 2 layers, `hidden_size` 48, a 382-token miniature of Granite's tokenizer, and random weights (85 kB). Its replies contain raw bytes that are not valid UTF-8, which tests incremental decoding. These cases need no download.
- The **real model**, `ibm-granite/granite-4.0-350m` (28 layers, 352 M weights, 705 MB), used for one or two cases per stage. These cases are skipped until you fetch the asset with `learnai assets fetch granite-4.0-350m`.

Every field of `config.json` differs between the two models, so always read `config` instead of hardcoding constants. Test cases are grouped by stage, and each stage only calls its own functions and those from earlier stages.

**Allowed and forbidden helpers.** Implement every component in `float32`. You may use `ops.rms_norm`, `ops.softmax`, `ops.top_k_filter`, `ops.top_p_filter`, `ops.min_p_filter`, and `learnai.kit.bpe.translate_pattern` from the kit. The harness rejects `transformers`, `tokenizers`, `safetensors`, `mlx_lm`, `jinja2`, the kit's `RealModel`, `RealLM`, `ByteLevelBPE`, and `SafetensorsFile`, MLX's built-in rotary embeddings (`nn.RoPE`, `mx.fast.rope`), and all random number generators.

## Stage 1: weights from a safetensors file

```python
def load_weights(file, names=None): ...
```

`file` is a kit `CountingFile` wrapping a read-only memory map of a safetensors file. Calling `len(file)` returns the file size at zero cost. Slicing `file[a:b]` returns a zero-copy `memoryview` of bytes `a` through `b - 1` and adds `b - a` to `file.counters.bytes_read`. Passing that slice to `np.frombuffer(file[a:b], dtype)` creates a NumPy view directly over the mapped pages.

**File format** (same as `safetensors-parse`). Bytes `0` to `7` store the header length $n$ as an unsigned 64-bit little-endian integer. Bytes `8` to $8 + n - 1$ store UTF-8 JSON (possibly padded with trailing spaces) mapping each tensor name to `{"dtype", "shape", "data_offsets": [begin, end]}`. If `"__metadata__"` appears in the header, ignore it because it is not a tensor. The **data section** starts at byte $8 + n$, and each tensor occupies bytes `begin` through `end - 1` of the data section as little-endian values in row-major order. Tensors may appear in any order in the data section and may have unused gaps between them.

**Behavior.** Return a `dict` mapping each requested tensor name to a `float32` tensor in your framework with the shape from the header (`[]` is a scalar):

| stored `dtype` | conversion to `float32` |
|---|---|
| `BF16` | Upper 16 bits of a `float32`. Shift each 16-bit pattern `b` left by 16 (`b << 16`) and reinterpret as `float32`. Exact. |
| `F16` | IEEE 754 half precision, cast to `float32`. Exact. |
| `F32` | Already `float32`. |

- When `names` is `None`, load every tensor with keys in header order. Otherwise `names` is a list of tensor names, and the returned dict must contain those exact keys in the same order.
- Raise `KeyError` if any requested name is missing from the header (including `"__metadata__"`). Perform this check before reading any tensor data.
- **Read budget.** The call may read at most the header plus the requested tensors: `file.counters.bytes_read` $\le 8 + n + \sum (\texttt{end} - \texttt{begin})$ over the requested names. Loading three tensors from a 705 MB file must only read those three tensors.

### Example 1

Consider a 234-byte file with header length $n = 200$ (199 bytes of JSON plus one trailing space):

```
{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
 "b":{"dtype":"BF16","shape":[2],"data_offsets":[16,20]},
 "h":{"dtype":"F16","shape":[3],"data_offsets":[20,26]}}
```

```python
load_weights(file)
# {"w": [[1., 2.], [3., 4.]], "b": [1., -2.5], "h": [0.5, -0.25, 8.]}  # all float32
load_weights(file, ["h"])  # {"h": [0.5, -0.25, 8.]}; reads 208 + 6 bytes
```

Tensor `b` sits at bytes $208 + 16$ through $208 + 19$ (`80 3f 20 c0`). Its 16-bit little-endian patterns are `0x3F80` and `0xC020`, which shift to `float32` bit patterns `0x3F800000` (`1.0`) and `0xC0200000` (`-2.5`). Decoding those same bits as IEEE `float16` would yield `1.875` and `-2.0625`.

## Stage 2: tokenizer and chat template

```python
class Tokenizer:
    def __init__(self, tokenizer_json): ...
    def encode(self, text): ...
    def decode(self, ids): ...
    def token_bytes(self, token): ...


def chat_prompt(messages): ...
```

`tokenizer_json` is the parsed `tokenizer.json` dict (same layout as `tokenizer-json-load`):

| JSON path | content |
|---|---|
| `["model"]["vocab"]` | `dict` mapping token string to integer id |
| `["model"]["merges"]` | list of `[left, right]` pairs; list index is the pair's **rank** |
| `["added_tokens"]` | list of dicts with `"content"`, `"id"`, and `"special"` (bool) |
| `["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]` | regex string that splits text into words |

**`encode(text)`** returns a list of Python `int` ids using four steps:

1. **Added tokens.** Scan `text` from left to right for the `"content"` of any added token, whether `"special"` is `True` or `False`. The leftmost match wins. If multiple added tokens start at the same position, the longest match wins. Replace each match with its `"id"`, and encode the remaining stretches between matches independently with steps 2 through 4.
2. **Pre-tokenization.** Split each stretch into consecutive matches of `re.compile(translate_pattern(pattern))`. BPE merges never cross word boundaries.
3. **Byte-to-character mapping.** Encode each word as UTF-8 and map each byte to one Unicode character. Bytes `0x21`–`0x7E`, `0xA1`–`0xAC`, and `0xAE`–`0xFF` map to the character with the same code point. The remaining 68 bytes map in increasing order to `U+0100`, `U+0101`, and so on (space `0x20` becomes `Ġ`, `U+0120`).
4. **BPE merges.** Start with the word split into single-character symbols. Repeatedly find the adjacent pair present in `merges` with the lowest rank, and replace every non-overlapping occurrence of that pair from left to right with its concatenation. Stop when no adjacent pair is in `merges`. Look up each final symbol in `vocab` to get its id (never compute ids from merge ranks).

Do not prepend or append special tokens automatically. `encode("")` returns `[]`.

**`token_bytes(token)`** returns the `bytes` represented by a single token id. For an added token, return its `"content"` encoded as UTF-8. For any other token, invert the byte-to-character mapping of step 3 on its vocabulary string.

**`decode(ids)`** concatenates `token_bytes(i)` for every id in `ids` (including special tokens) and decodes the resulting byte string as UTF-8 with `errors="replace"` (invalid sequences become `U+FFFD`). Separate `Tokenizer` instances must share no mutable state.

**`chat_prompt(messages)`** formats a non-empty list of dicts (`{"role": ..., "content": ...}` where `role` is `"system"`, `"user"`, or `"assistant"`) in Granite's chat format (same as `real-model-chat-turn`). Each turn uses the template:

```
<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n
```

Concatenate three parts:

1. **System turn (always present).** If the first message has role `"system"` and non-empty `"content"`, use that content and consume the message. Otherwise use the default text `You are a helpful assistant. Please ensure responses are professional, accurate, and safe.` (if the first message is `"system"` with empty content, drop it in favor of this default).
2. **Remaining messages.** Render each remaining message in order with its own role and verbatim content, even if the content is empty or a later message has role `"system"`.
3. **Generation prompt.** Append `<|start_of_role|>assistant<|end_of_role|>` with no trailing newline.

Encoding `chat_prompt(messages)` with `Tokenizer.encode` turns each of the three role/end markers into a single added-token id.

### Example 2

Using the tiny model's tokenizer (382 ids: 256 byte tokens, 120 merges, and 6 added tokens, including `<|start_of_role|>` = 377, `<|end_of_role|>` = 378, and `<|end_of_text|>` = 379):

```python
tok = Tokenizer(tokenizer_json)
tok.encode("Hello there! How can I help you today?")
# [39, 283, 288, 267, 261, 0, 273, 296, 265, 256, 374, 322, 220, 373, 260, 292, 339, 30]
tok.token_bytes(283)  # b"el"
tok.token_bytes(267)  # b" the"
tok.decode([172, 253])  # "\ufffd" (first two bytes of a 4-byte emoji)

chat_prompt([{"role": "user", "content": "Hi!"}])
```

```
<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Please ensure responses are professional, accurate, and safe.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Hi!<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
```

That prompt encodes to 54 ids with the tiny tokenizer and 32 ids with the real tokenizer. On the real tokenizer, `"In 2026"` encodes to `[644, 220, 2366, 21]` because the regex splits digits into groups of up to three and keeps a space before a digit as its own word.

## Stage 3: forward pass

```python
def forward(weights, config, ids): ...
```

`weights` is the dict returned by `load_weights` for the full checkpoint, and `config` is the parsed `config.json`. Let `head_dim = hidden_size // num_attention_heads`. Read all architecture parameters from `config`:

| `config` key | tiny | Granite 4.0 350M | symbol / role |
|---|---|---|---|
| `num_hidden_layers` | 2 | 28 | number of layers $L$ |
| `hidden_size` | 48 | 1024 | model width $d$ |
| `num_attention_heads` | 4 | 16 | query heads $H$ |
| `num_key_value_heads` | 2 | 4 | key/value heads $H_{kv}$ |
| `shared_intermediate_size` | 32 | 2048 | MLP hidden width `d_ff` |
| `embedding_multiplier` | 6.0 | 12 | scales token embeddings |
| `residual_multiplier` | 0.4 | 0.263 | scales both residual branches |
| `attention_multiplier` | 0.125 | 0.015625 | scales attention scores |
| `logits_scaling` | 0.5 | 4 | divides final logits |
| `rope_theta` | 10000.0 | 10000000 | RoPE base frequency |
| `rms_norm_eps` | 1e-05 | 1e-05 | RMSNorm epsilon |
| `vocab_size` | 382 | 100352 | vocabulary size $V$ |
| `eos_token_id` | 379 | 100257 | end-of-sequence id (stage 4) |

**Forward pass.** Given token ids `ids` of shape `(T,)` ($T \ge 1$) at positions $p = 0, 1, \dots, T - 1$, let $E$ be `model.embed_tokens.weight`. No projection matrix has a bias, and every linear weight is stored in `(out, in)` layout.

1. **Embedding.** Compute $x = E[\text{ids}] \cdot \texttt{embedding\_multiplier}$. There is no learned position embedding.
2. **Transformer layers.** For $i = 0, \dots, L - 1$, use weight prefix `model.layers.{i}.` and residual multiplier $r = \texttt{residual\_multiplier}$:
   $$x \leftarrow x + r \cdot \operatorname{Attn}_i(\operatorname{RMSNorm}(x))$$
   $$x \leftarrow x + r \cdot \operatorname{MLP}_i(\operatorname{RMSNorm}(x))$$
   - **RMSNorm.** For input $u$ and gain $g$, compute $u / \sqrt{\operatorname{mean}(u^2) + \texttt{rms\_norm\_eps}} \cdot g$ over the last axis. Use `input_layernorm.weight` before attention and `post_attention_layernorm.weight` before the MLP.
   - **Grouped-query attention.** Project queries with `self_attn.q_proj.weight` into $H$ heads, and project keys and values with `k_proj` and `v_proj` into $H_{kv}$ heads, each of width `head_dim`. Query head $h$ pairs with key/value head $\lfloor h / (H / H_{kv}) \rfloor$.
   - **RoPE (half-split).** Rotate queries and keys (not values). Feature $j$ pairs with feature $j + \texttt{head\_dim}/2$ for $0 \le j < \texttt{head\_dim}/2$. At position $p$, the rotation angle for pair $j$ is $\phi_{p,j} = p \cdot \texttt{rope\_theta}^{-2j/\texttt{head\_dim}}$, transforming $(a, b)$ to $(a \cos\phi_{p,j} - b \sin\phi_{p,j},\; a \sin\phi_{p,j} + b \cos\phi_{p,j})$.
   - **Attention output.** Scale dot-product scores by `attention_multiplier`, apply a causal mask so position $p$ attends only to positions $\le p$, take softmax over keys, compute the weighted sum of values, concatenate the $H$ heads, and project with `self_attn.o_proj.weight`.
   - **SwiGLU MLP.** For normalized input $u$, compute $z = u W_{\text{in}}^\top$ using `shared_mlp.input_linear.weight` (`2 * d_ff` rows). Split $z$ along its last axis into `gate` (first `d_ff` features) and `up` (remaining `d_ff` features). Return $(\operatorname{silu}(\text{gate}) \odot \text{up}) W_{\text{out}}^\top$ using `shared_mlp.output_linear.weight`.
3. **Output head.** Apply `RMSNorm` with `model.norm.weight`, then compute logits as $x E^\top / \texttt{logits\_scaling}$. The output head shares weights with the embedding table (`"tie_word_embeddings": true`; there is no `lm_head.weight`).

`forward(weights, config, ids)` returns a `float32` tensor of shape `(T, V)`.

### Example 3

On the 18 ids from Example 2, the tiny model returns logits of shape `(18, 382)`. Row 17 peaks at about `6.27` on token `325` (` capital`). On the real model, the 12 tokens of `"A B C D E F G H I J K L"` produce a `(12, 100352)` tensor whose final row peaks at the token for `" M"`.

## Stage 4: cached generation and sampling

```python
def prefill(weights, config, prompt_ids): ...  # -> (logits, cache)
def decode_step(weights, config, token, cache): ...  # -> (logits, cache)
def sample_token(logits, sampler, history, u): ...  # -> int
def generate(weights, config, prompt_ids, max_new_tokens, sampler, uniforms): ...  # -> list[int]
```

**Cache layout.** Tests inspect your caches and pass their own caches into `decode_step`, so use this exact dictionary structure:

```python
{"length": n, "keys": [K_0, K_1, ...], "values": [V_0, V_1, ...]}
```

- `length` is a Python `int` giving the number of cached tokens $n$.
- `keys` and `values` each hold one `float32` tensor per layer of shape `(num_key_value_heads, C, head_dim)` with capacity $C \ge n$.
- Slot $s < n$ of `K_i[h]` holds the key for position $s$ in key/value head $h$ of layer $i$, **already rotated** by RoPE for position $s$. `V_i` holds the corresponding values. Slots at indices $\ge n$ are unused. Store keys and values at key/value-head width, not expanded to query-head width.

**`prefill(weights, config, prompt_ids)`** processes the full prompt (`prompt_ids` of shape `(P,)`) in one pass. It returns `(logits, cache)`, where `logits` has shape `(V,)` for the **last** prompt position and `cache` has `length == P`.

**`decode_step(weights, config, token, cache)`** accepts a single Python `int` token at position $n$ and a cache of length $n$ (with any capacity $C \ge n$). It returns `(logits, new_cache)` where `logits` has shape `(V,)` and matches running `forward` on the full $n + 1$ tokens, and `new_cache` has `length == n + 1`. **Do not mutate the input `cache`** or share a mutable buffer that overwrites its first $n$ slots. Callers may branch from the same cache multiple times.

**`sample_token(logits, sampler, history, u)`** selects one Python `int` token from a `(V,)` logit vector `logits` (written $x$ below) using the `sampler-chain` pipeline. `sampler` is a dict with these exact keys:

| key | valid range | no-op value |
|---|---|---|
| `"repetition_penalty"` | $\ge 1$ | `1.0` |
| `"frequency_penalty"` | $\ge 0$ | `0.0` |
| `"presence_penalty"` | $\ge 0$ | `0.0` |
| `"top_k"` | int $\ge 0$ | `0` or any value $\ge V$ |
| `"top_p"` | $0 < p \le 1$ | `1.0` |
| `"min_p"` | $0 \le m < 1$ | `0.0` |
| `"temperature"` | $> 0$ | `1.0` |

Removing a token sets its logit to `-inf`. Break ties in logit or probability order by lowest token id first. Existing `-inf` entries remain `-inf`.

1. **Penalties.** Let $c_v$ be the count of token $v$ in `history` (a list of ints), $\theta$ be `repetition_penalty`, $\alpha_f$ be `frequency_penalty`, and $\alpha_p$ be `presence_penalty`. For every token with $c_v > 0$, apply repetition penalty first: $x_v \leftarrow x_v / \theta$ if $x_v > 0$, else $x_v \cdot \theta$. Then apply frequency and presence penalties: $x_v \leftarrow x_v - c_v \alpha_f - \alpha_p$.
2. **Top-k.** Keep the `top_k` largest logits and set the rest to `-inf`.
3. **Top-p.** Compute softmax probabilities on the surviving logits and sort tokens by decreasing probability. Keep a token if and only if the cumulative probability strictly before it is less than `top_p`. Skip this stage when `top_p == 1.0`.
4. **Min-p.** Recompute softmax probabilities on the surviving logits and remove any token whose probability is strictly less than `min_p` times the maximum probability.
5. **Temperature.** Divide all logits by `temperature`.
6. **Inverse-CDF draw.** Let $\pi = \operatorname{softmax}(x)$ and $c_k = \sum_{i=0}^{k} \pi_i$. Return the smallest index $k$ such that $c_k > u \cdot c_{V-1}$, where `u` is a caller-supplied float in $[0, 1)$. Setting `top_k = 1` always selects the greedy token regardless of `u`.

**`generate(weights, config, prompt_ids, max_new_tokens, sampler, uniforms)`** runs `prefill`, then loops for `step` in `0..max_new_tokens - 1`. At each step, call `sample_token(logits, sampler, history, uniforms[step])`, where `history` contains only the tokens generated so far in this call (not `prompt_ids`). If the sampled token equals `config["eos_token_id"]`, stop immediately without appending it or calling `decode_step`. Otherwise append the token to `history`, and if more steps remain, call `decode_step` to get the next logits. `uniforms` is a list of `max_new_tokens` floats. Return `history` as a list of Python `int`s.

### Example 4

Call `sample_token` with `logits = [3.0, 2.0, 4.0, 0.0, 1.0, 2.5]`, `history = [2, 2, 3]`, and `{"repetition_penalty": 2.0, "frequency_penalty": 0.0, "presence_penalty": 0.5, "top_k": 4, "top_p": 0.85, "min_p": 0.4, "temperature": 2.0}`:

| stage | logits afterwards | note |
|---|---|---|
| penalties | `[3, 2, 1.5, -0.5, 1, 2.5]` | token 2 gets $4 / 2 - 0.5$; token 3 gets $0 \cdot 2 - 0.5$ |
| top-k 4 | `[3, 2, 1.5, -inf, -inf, 2.5]` | |
| top-p 0.85 | `[3, 2, -inf, -inf, -inf, 2.5]` | probabilities are 0.455, 0.276, 0.167, 0.102 for tokens 0, 5, 1, 2; mass before token 2 is 0.898 |
| min-p 0.4 | `[3, -inf, -inf, -inf, -inf, 2.5]` | renormalized probabilities are 0.507, 0.307, 0.186; threshold is $0.4 \cdot 0.507$ |
| temperature 2 | `[1.5, -inf, -inf, -inf, -inf, 1.25]` | $\pi$ is 0.562 for token 0 and 0.438 for token 5 |

With these probabilities, `u = 0.5` returns token `0` and `u = 0.6` returns token `5`. Applying temperature before the filters would have kept token `2` in top-p and token `1` in min-p.

On the real model, the 32-token prompt for `[{"role": "user", "content": "Hi!"}]` with `top_k = 40`, `top_p = 0.9`, `min_p = 0.02`, `temperature = 0.8`, and penalties `(1.15, 0.2, 0.1)` generates nine tokens (`Hello! How can I assist you today?`) for the fixture's ten uniforms, then draws `eos_token_id` on the tenth step.

## Stage 5: streaming text and stop strings

```python
def stream_chat(weights, config, tokenizer, messages, max_new_tokens, sampler, uniforms, stop): ...
```

`stream_chat` is a generator that yields non-empty `str` pieces of the reply. `tokenizer` is an instance of your `Tokenizer`, and `stop` is a list of non-empty stop strings (possibly empty).

1. **Token stream.** Encode `chat_prompt(messages)` with `tokenizer.encode` and generate tokens using the same logic as `generate`.
2. **Incremental UTF-8 decoding.** Feed each generated token's bytes into `codecs.getincrementaldecoder("utf-8")(errors="replace")`. Tokens marked `"special": true` in `tokenizer.json` contribute `b""`, while non-special added tokens contribute their UTF-8 content. After $i$ tokens, let $t_i$ be the accumulated decoded string. Invalid byte sequences become `U+FFFD`, while an incomplete multi-byte character at the end waits for subsequent tokens.
3. **Stop matching and lookahead holdback.** A *match* is any occurrence of a string from `stop` inside $t_i$.
   - If $t_i$ contains one or more matches, choose the match that **ends first** (breaking ties by the one that starts first). The visible text is $t_i$ up to the start of that match. Yield any unyielded visible text and stop immediately without generating further tokens.
   - Otherwise, find the longest suffix of $t_i$ that is a **proper prefix** of any stop string, and hold it back because future tokens could complete that stop string. The visible text is everything in $t_i$ before that suffix.
4. **Yielding.** After each token, yield the newly visible slice of text if it is non-empty.
5. **End of generation.** If generation ends via `eos_token_id` or `max_new_tokens` without a match, flush the decoder with `decode(b"", final=True)` so any trailing incomplete character becomes `U+FFFD`. Check once more for a stop match in the finalized text. If a match exists, cut at the match as in step 3; otherwise release all remaining text including any held suffix. Yield whatever has not yet been yielded, if non-empty.

Concatenating all yielded pieces reproduces the reply up to the first stop string. At most one piece is yielded per token, plus at most one final piece at the end.

### Example 5

On the tiny model, replying to `[{"role": "user", "content": "Hi!"}]` with `stop = ["ro\r"]` (`\r` is carriage return), `top_k = 12`, `temperature = 1.6`, `repetition_penalty = 1.3`, and `max_new_tokens = 20`:

| token | its bytes | yielded | why |
|---|---|---|---|
| 1 | `or` | `"o"` | `r` could start `"ro\r"`, so hold it |
| 2 | `96` (hex) | `"r\ufffd"` | stray continuation byte; `r` was not a stop |
| 3 | `16` (hex) | `"\x16"` | |
| 4 | `(` | `"("` | |
| 5 | `st` | `"st"` | |
| 6 | `F0` (hex) | nothing | leading byte of a 4-byte UTF-8 character |
| 7 | `14` (hex) | `"\ufffd\x14"` | incomplete sequence flushed as `U+FFFD` |
| 8 | `ser` | `"se"` | hold `r` again |
| 9 | `en` | `"ren"` | |
| 10 | `E` | `"E"` | |
| 11 | ` pro` | `" p"` | hold `ro` |
| 12 | `\r` | nothing | `"ro\r"` completed; generation stops |

Only 12 of the 20 tokens are generated. With `stop = []` and the same uniforms, all 20 tokens are generated. Token 17 is the special token `<|unk|>`, which is fed back into `decode_step` but emits no bytes.

On the real model, `system: "Be brief."` and `user: "Count from one to ten in words."` with `top_k = 1` begins `One, two, three, four, ...`. With `stop = ["\n\n", ", three"]`, `stream_chat` yields `"One"` and `", two"`. Each comma is held until the next token proves ` three` does not follow, and the fifth token completes the stop string.

## Shapes

| name | shape | dtype |
|---|---|---|
| values of `load_weights` | from file header | `float32` |
| `ids`, `prompt_ids` | `(T,)`, `(P,)` | integer |
| output of `forward` | `(T, V)` | `float32` |
| logits of `prefill`, `decode_step`; input to `sample_token` | `(V,)` | `float32` |
| `cache["keys"][i]`, `cache["values"][i]` | `(H_kv, C, head_dim)`, $C \ge$ `cache["length"]` | `float32` |
| `model.embed_tokens.weight` | `(V, d)` | `float32` |
| `self_attn.q_proj.weight`, `self_attn.o_proj.weight` | `(d, d)` | `float32` |
| `self_attn.k_proj.weight`, `self_attn.v_proj.weight` | `(H_kv * head_dim, d)` | `float32` |
| `shared_mlp.input_linear.weight` | `(2 * d_ff, d)` | `float32` |
| `shared_mlp.output_linear.weight` | `(d, d_ff)` | `float32` |
| `input_layernorm.weight`, `post_attention_layernorm.weight`, `model.norm.weight` | `(d,)` | `float32` |

## Constraints

- **Reference oracle.** Fixtures were generated from the kit's reference implementations and verified against `transformers` (logits within `7e-5` on the real model) and `tokenizers`.
- **Exact checks.** Weights (`BF16` and `F16` convert to `float32` without rounding), token ids, prompt strings, decoded strings, sampled tokens, and the list of streamed pieces must match exactly. `bytes_read` must satisfy the stage 1 bound with no slack.
- **Numeric tolerances.** Logits, cached keys, and cached values are checked with `rtol = 1e-3` and `atol = 1e-3` (on the real model, across 64 fixed vocabulary columns and 3 cache layers). Float32 implementations across NumPy, PyTorch, JAX, and MLX agree to about `1e-4` on logits of magnitude 25.
- **Deterministic sampling.** Every test uniform was selected so its sampled token survives a `3e-3` perturbation to logits and a `1e-4` perturbation to `u`. Floating-point noise will not flip a token, whereas wrong stage ordering will.
- Prompts have at most 65 tokens and generation produces at most 20 tokens. The harness calls your `load_weights` and `Tokenizer` once per run on the real model and reuses them across later stages.
- Time limit is 30 seconds per case. The reference takes about 3 seconds total in PyTorch and MLX, and about 8 seconds in JAX (15 seconds on first run during compilation).

> [!NOTE]
> JAX compiles each operation once per distinct input shape. Growing a cache by one slot per token triggers a recompile on every decode step. Use the cache capacity $C$ to allocate buffers in larger chunks (for example, multiples of 32), mask unused slots, and update with `.at[]`. In PyTorch and MLX, concatenating along the sequence axis with $C = n$ is fast enough.

## Why this matters

Every production inference runtime (`llama.cpp`, `mlx-lm`, `transformers`) is this exact five-stage pipeline plus performance optimizations. Nearly every bug in this pipeline fails silently while still producing plausible output. Counting safetensors offsets from byte 0 fills valid-shaped tensors with header bytes, and decoding `BF16` as `F16` produces small, believable weights. Skipping pre-tokenization or sending the wrong chat template still yields fluent text, only slightly worse. Using `1 / sqrt(head_dim)` instead of `attention_multiplier`, omitting `residual_multiplier` on one branch, or swapping the fused MLP halves all produce finite logits. Rotating new keys at position 0 only degrades longer generations, applying temperature before top-p changes which tokens survive, and yielding a stop-string prefix cannot be undone once streamed to a client.
