Build everything between three files on disk and a streamed chat reply: a runtime
for IBM's Granite 4.0 350M that reads its weights from `model.safetensors` through a
memory map, tokenizes with its `tokenizer.json`, renders its chat layout, runs its
forward pass, generates with a key/value cache and a sampler chain, and streams
text that stops at a stop string. Every component is one you have written before in
its own problem. Here they are one file with no library underneath, and each stage
feeds the next: the weights your loader returns are the weights your forward pass
runs on.

## Overview

```python
def load_weights(file, names=None): ...  # stage 1 -> {name: float32 tensor}


class Tokenizer:  # stage 2
    def __init__(self, tokenizer_json): ...
    def encode(self, text): ...  # -> list of ints
    def decode(self, ids): ...  # -> str
    def token_bytes(self, token): ...  # -> bytes


def chat_prompt(messages): ...  # stage 2 -> str
def forward(weights, config, ids): ...  # stage 3 -> logits (T, V)
def prefill(weights, config, prompt_ids): ...  # stage 4 -> (logits (V,), cache)
def decode_step(weights, config, token, cache): ...  # stage 4 -> (logits (V,), cache)
def sample_token(logits, sampler, history, u): ...  # stage 4 -> int
def generate(weights, config, prompt_ids, max_new_tokens, sampler, uniforms): ...  # -> list
def stream_chat(
    weights, config, tokenizer, messages, max_new_tokens, sampler, uniforms, stop
): ...  # stage 5: a generator of str
```

![The runtime. Stage 1 turns the checkpoint into float32 tensors, stage 2 turns a conversation into token ids, stage 3 is the forward pass, stage 4 wraps it in a cache and a sampler, and stage 5 turns the sampled tokens into text that is held back while it may still become a stop string.](assets/architecture.svg)

**Two models.** Every stage is tested on two model directories with the same three
files, `config.json`, `tokenizer.json`, and `model.safetensors`:

- a **tiny model** that ships with the problem (`fixtures/tiny/`): two layers,
  `hidden_size` 48, a 382-token miniature of Granite's tokenizer, random weights,
  85 kB. Its text is noise, which is useful: replies are full of bytes that are not
  UTF-8. Its cases need no download.
- the **real model**, `ibm-granite/granite-4.0-350m` (28 layers, 352 M weights,
  705 MB), for one or two cases per stage. They are skipped until the asset is
  fetched (`ws assets fetch granite-4.0-350m`).

The two configs have **different numbers** in every field, so read `config`; do not
write Granite's constants into the code. The cases are grouped by stage, and a
stage's cases call only that stage's functions and those of earlier stages.

**What is yours and what is given.** Write every component yourself, in float32.
From the kit, `ops.rms_norm`, `ops.softmax`, the three sampler filters
(`ops.top_k_filter`, `ops.top_p_filter`, `ops.min_p_filter`), and
`whetstone.kit.bpe.translate_pattern` may be used, as in the earlier problems. Not
allowed, and rejected by the harness: the `transformers`, `tokenizers`,
`safetensors`, `mlx_lm`, and `jinja2` packages; the kit's `RealModel`, `RealLM`,
`ByteLevelBPE`, and `SafetensorsFile`; MLX's built-in rotary embedding (`nn.RoPE`,
`mx.fast.rope`); and every random number generator.

## Stage 1: weights from a safetensors file

```python
def load_weights(file, names=None): ...
```

`file` is a kit `CountingFile` opened on a safetensors file: a read-only memory map
that counts what is read from it. `len(file)` is the file's size and costs nothing;
`file[a:b]` is a zero-copy `memoryview` of bytes `a` to `b - 1` and counts as reading
`b - a` bytes, so `np.frombuffer(file[a:b], dtype)` is an array over the file's pages.

**The format** (as in `safetensors-parse`). Bytes 0 to 7 hold the header length $n$,
an unsigned 64-bit little-endian integer. Bytes 8 to $8 + n - 1$ are UTF-8 JSON, one
object that maps each tensor name to `{"dtype", "shape", "data_offsets": [begin,
end]}`; the header may end in spaces; the key `"__metadata__"`, if present, is not a
tensor. A tensor occupies bytes `begin` to `end - 1` of the **data section**, which
starts at byte $8 + n$ of the file, as little-endian elements in row-major order.
Tensors need not lie in header order, and there may be unused bytes between them.

**`load_weights(file, names=None)`** returns a dict from tensor name to a **float32
tensor of your framework** with the shape given in the header (`[]` is a scalar):

| stored `dtype` | value |
|---|---|
| `BF16` | bfloat16, the upper 16 bits of a float32: the pattern `b` is the float32 whose bits are `b << 16`. Exact |
| `F16` | IEEE half precision, converted to float32. Exact |
| `F32` | as stored |

- `names=None` loads every tensor, with the dict's keys in header order. Otherwise
  `names` is a list of tensor names, and the dict has exactly those keys, in that
  order.
- `KeyError` if a name is not a tensor of the file (`"__metadata__"` is not one),
  raised **before any tensor data is read**.
- **Cost.** The call may read at most $8 + n$ bytes plus the bytes of the tensors
  it returns: `file.counters.bytes_read` $\le 8 + n + \sum (\texttt{end} -
  \texttt{begin})$ over the requested names. Loading three tensors of a 705 MB file
  reads those three tensors.

### Example 1

A file of 234 bytes. Its header is $n = 200$ bytes: 199 bytes of JSON and one space.

```
{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
 "b":{"dtype":"BF16","shape":[2],"data_offsets":[16,20]},
 "h":{"dtype":"F16","shape":[3],"data_offsets":[20,26]}}
```

```python
load_weights(file)
# {"w": [[1., 2.], [3., 4.]], "b": [1., -2.5], "h": [0.5, -0.25, 8.]}   all float32
load_weights(file, ["h"])  # {"h": [0.5, -0.25, 8.]}; reads 208 + 6 bytes
```

`b` is at bytes $208 + 16$ to $208 + 19$ of the file: `80 3f 20 c0`, the 16-bit
patterns `0x3F80` and `0xC020`, which are the float32 values `0x3F800000` and
`0xC0200000`. Read as IEEE half precision, the same bits would be 1.875 and −2.0625.

## Stage 2: tokenizer and chat template

```python
class Tokenizer:
    def __init__(self, tokenizer_json): ...
    def encode(self, text): ...
    def decode(self, ids): ...
    def token_bytes(self, token): ...


def chat_prompt(messages): ...
```

`tokenizer_json` is the parsed `tokenizer.json`. The parts that matter, as in
`tokenizer-json-load`:

| path in the JSON | content |
|---|---|
| `["model"]["vocab"]` | dict from token string to id |
| `["model"]["merges"]` | list of pairs `[left, right]`; the position in the list is the pair's **rank** |
| `["added_tokens"]` | list of objects with `"content"`, `"id"`, and `"special"` (a bool) |
| `["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]` | the regular expression that cuts text into words |

**`encode(text)`** returns a list of Python ints:

1. **Added tokens.** Scan the text from left to right for the `content` of any added
   token, special or not. The leftmost occurrence wins; among added tokens that
   start at the same position, the longest. Each occurrence becomes its `id`. The
   stretches between occurrences are encoded by steps 2 to 4, each on its own.
2. **Words.** Cut the stretch into the successive matches of the pattern, compiled
   as `re.compile(translate_pattern(pattern))`. Merges never cross a word boundary.
3. **Bytes as characters.** Encode the word as UTF-8 and write each byte as one
   character: the bytes `0x21`–`0x7E`, `0xA1`–`0xAC`, and `0xAE`–`0xFF` as the
   character with the same code point, and the other 68 bytes, in increasing order,
   as U+0100, U+0101, and so on (the space is `Ġ`, U+0120).
4. **BPE.** Start with the word as single characters. Repeat: among the adjacent
   pairs that are in `merges`, take the one of lowest rank and replace every
   occurrence of it, left to right and without overlap, by the concatenation. Stop
   when no adjacent pair is in `merges`. The ids are `vocab[s]` for the strings that
   remain. Ids are looked up, never computed from ranks.

Nothing is added around the text. `encode("")` is `[]`.

**`token_bytes(token)`** returns the `bytes` of one token: for an added token its
`content` in UTF-8, for any other token its vocabulary string read back through the
alphabet of step 3. **`decode(ids)`** joins the bytes of all tokens, special or not,
and decodes them as UTF-8 with invalid sequences replaced by U+FFFD
(`errors="replace"`). Two tokenizers built from two files share no state.

**`chat_prompt(messages)`** renders a conversation, a non-empty list of dicts with
`"role"` (`"system"`, `"user"`, or `"assistant"`) and `"content"` (a string), in
Granite's layout, as in `real-model-chat-turn`. Each turn is

```
<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n
```

and the prompt is the concatenation of:

1. **A system turn, always.** If the first message has the role `system` and its
   content is not empty, that content is the system text and the message is not
   rendered again. Otherwise the system text is `You are a helpful assistant. Please
   ensure responses are professional, accurate, and safe.` (a first system message
   with empty content is dropped in favor of it).
2. **One turn per remaining message**, in order, with its own role (a later `system`
   message is an ordinary turn). Content is inserted verbatim, also when empty.
3. **The generation prompt** `<|start_of_role|>assistant<|end_of_role|>`, with
   nothing after it.

The prompt ids of a conversation are `tokenizer.encode(chat_prompt(messages))`; the
three markers are added tokens and become one id each.

### Example 2

With the tiny model's tokenizer (382 ids: the 256 bytes, 120 merges, and six added
tokens, among them `<|start_of_role|>` 377, `<|end_of_role|>` 378, and
`<|end_of_text|>` 379):

```python
tok = Tokenizer(tokenizer_json)
tok.encode("Hello there! How can I help you today?")
# [39, 283, 288, 267, 261, 0, 273, 296, 265, 256, 374, 322, 220, 373, 260, 292, 339, 30]
tok.token_bytes(283)  # b"el"
tok.token_bytes(267)  # b" the"
tok.decode([172, 253])  # "\ufffd": the first two of the four bytes of an emoji

chat_prompt([{"role": "user", "content": "Hi!"}])
```

```
<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Please ensure responses are professional, accurate, and safe.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Hi!<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
```

That prompt is 54 ids with the tiny tokenizer and 32 with the real one, where
`"In 2026"` encodes to `[644, 220, 2366, 21]`: the pattern cuts digit runs into
groups of three and gives a space before a digit a word of its own.

## Stage 3: forward pass

```python
def forward(weights, config, ids): ...
```

`weights` is the dict your `load_weights` returned for the whole checkpoint, and
`config` the parsed `config.json`. The fields to read, with $H$ =
`num_attention_heads` and `head_dim` = `hidden_size / H`:

| `config` key | tiny | Granite 4.0 350M | used as |
|---|---|---|---|
| `num_hidden_layers` | 2 | 28 | number of blocks |
| `hidden_size` | 48 | 1024 | $d$ |
| `num_attention_heads` | 4 | 16 | query heads $H$ |
| `num_key_value_heads` | 2 | 4 | key/value heads $H_{kv}$ |
| `shared_intermediate_size` | 32 | 2048 | `d_ff` |
| `embedding_multiplier` | 6.0 | 12 | scales the embedding |
| `residual_multiplier` | 0.4 | 0.263 | scales both branches of a block |
| `attention_multiplier` | 0.125 | 0.015625 | scales the attention scores |
| `logits_scaling` | 0.5 | 4 | divides the logits |
| `rope_theta` | 10000.0 | 10000000 | RoPE base |
| `rms_norm_eps` | 1e-05 | 1e-05 | epsilon of every RMSNorm |
| `vocab_size` | 382 | 100352 | $V$ |
| `eos_token_id` | 379 | 100257 | end-of-sequence (stage 4) |

**The model.** For tokens at positions $0, 1, 2, \dots$, with
$E$ = `model.embed_tokens.weight`:

1. $x = E[\text{ids}] \cdot \texttt{embedding\_multiplier}$. There is no position
   embedding.
2. For layer $i = 0, 1, \dots$ with the prefix `model.layers.{i}.` and
   $r$ = `residual_multiplier`:
   $x \leftarrow x + r \cdot \operatorname{Attn}_i(\operatorname{RMSNorm}(x))$, then
   $x \leftarrow x + r \cdot \operatorname{MLP}_i(\operatorname{RMSNorm}(x))$.
   - RMSNorm is $u / \sqrt{\operatorname{mean}(u^2) + \texttt{rms\_norm\_eps}} \cdot
     g$ over the last axis, with the gains `input_layernorm.weight` (before
     attention) and `post_attention_layernorm.weight` (before the MLP).
   - Attention: queries from `self_attn.q_proj.weight` are split into $H$ heads,
     keys and values from `k_proj` and `v_proj` into $H_{kv}$ heads, all of width
     `head_dim`. Query head $h$ uses key/value head $\lfloor h / (H / H_{kv})
     \rfloor$. Queries and keys (not values) are rotated with RoPE in the
     **half-split** layout: feature $j$ of a head pairs with feature
     $j + \texttt{head\_dim}/2$, the angle of pair $j$ at position $p$ is
     $p \cdot \texttt{rope\_theta}^{-2j/\texttt{head\_dim}}$, and $(a, b)$ becomes
     $(a \cos - b \sin,\; a \sin + b \cos)$. The scores are $q\,k^\top \cdot
     \texttt{attention\_multiplier}$; position $p$ attends to positions $\le p$;
     softmax; the weighted sum of the values; heads concatenated; `o_proj`. No
     matrix has a bias, and all are stored `(out, in)`.
   - MLP: $z = u\,W_{\text{in}}^\top$ with `shared_mlp.input_linear.weight`, which
     has `2 * d_ff` rows. The first `d_ff` features of $z$ are the gate, the rest
     are up; the output is $(\operatorname{silu}(\text{gate}) \odot
     \text{up})\,W_{\text{out}}^\top$ with `shared_mlp.output_linear.weight`.
3. $x \leftarrow \operatorname{RMSNorm}(x)$ with `model.norm.weight`; the logits are
   $x\,E^\top / \texttt{logits\_scaling}$. The output head is the embedding matrix
   itself (both configs say `"tie_word_embeddings": true`, and neither checkpoint
   has an `lm_head.weight`).

`forward(weights, config, ids)` takes an integer tensor of shape `(T,)`, $T \ge 1$,
and returns the logits of every position, a float32 tensor of shape `(T, V)`.

### Example 3

For the 18 ids of Example 2, the tiny model's logits have shape `(18, 382)`; row 17
has its largest entry, about 6.27, at token 325 (` capital`). On the real model, the 12 tokens of
`"A B C D E F G H I J K L"` give a `(12, 100352)` tensor whose last row has its
maximum at the token of `" M"`.

## Stage 4: cached generation and sampling

```python
def prefill(weights, config, prompt_ids): ...  # -> (logits, cache)
def decode_step(weights, config, token, cache): ...  # -> (logits, cache)
def sample_token(logits, sampler, history, u): ...  # -> int
def generate(weights, config, prompt_ids, max_new_tokens, sampler, uniforms): ...  # -> list
```

**The cache** has a fixed layout, because the tests build caches and read yours:

```python
{"length": n, "keys": [K_0, K_1, ...], "values": [V_0, V_1, ...]}
```

- `length` is a Python int: the number of tokens the cache holds.
- `keys` and `values` are lists with one float32 tensor per layer, of shape
  `(num_key_value_heads, C, head_dim)` with capacity $C \ge n$. Slot $s < n$ of
  `K_i[h]` is the key of the token at position $s$ in key/value head $h$ of layer
  $i$, **already rotated** for position $s$; `V_i` holds the values. Slots from $n$
  on are unused and may contain anything. Keys and values are stored at the width
  of the key/value heads, not expanded to the query heads.

**`prefill(weights, config, prompt_ids)`** runs the model on the whole prompt, an
integer tensor of shape `(P,)`, in one pass. It returns `(logits, cache)`: the logits
of the **last** position, shape `(V,)`, and a cache of length $P$.

**`decode_step(weights, config, token, cache)`** takes one more token, a Python
`int`, and a cache of length $n$ (yours, or one a test built, with any capacity
$\ge n$). The token is at position $n$. It returns the logits at position $n$, shape
`(V,)`, equal to what the model computes for the whole sequence of $n + 1$ tokens,
and a cache of length $n + 1$. Nothing but the cache tells it what came before.
**It must not change the cache it is given**: neither `length` nor any of the first
$n$ slots, also not through a buffer that the old and the new cache share. A caller
may continue from one cache twice, and continue either result.

**`sample_token(logits, sampler, history, u)`** turns one row of logits, shape
`(V,)`, into a token (a Python `int`) by the chain of `sampler-chain`. `sampler` is a
dict with exactly these keys:

| key | range | value that switches the stage off |
|---|---|---|
| `"repetition_penalty"` | $\ge 1$ | `1.0` |
| `"frequency_penalty"` | $\ge 0$ | `0.0` |
| `"presence_penalty"` | $\ge 0$ | `0.0` |
| `"top_k"` | int $\ge 0$ | `0`, or any value $\ge V$ |
| `"top_p"` | $0 < p \le 1$ | `1.0` |
| `"min_p"` | $0 \le m < 1$ | `0.0` |
| `"temperature"` | $> 0$ | `1.0` |

With $x$ the logits, "removing" a token means setting its logit to `-inf`, and equal
values are ranked by token id, lowest first:

1. **Penalties.** Let $c_v$ be the number of times token $v$ occurs in `history`, a
   list of ints. For every token with $c_v > 0$: $x_v \leftarrow x_v / \theta$ if
   $x_v > 0$, else $x_v \cdot \theta$ (repetition penalty $\theta$); then
   $x_v \leftarrow x_v - c_v\,\alpha_f - \alpha_p$ (frequency and presence).
2. **Top-k.** Keep the `top_k` largest logits and remove the rest.
3. **Top-p.** Take the softmax of the current logits. Rank the tokens by decreasing
   probability and keep a token exactly when the probability ranked strictly before
   it is below `top_p`. With `top_p = 1.0` there is no top-p stage.
4. **Min-p.** Take the softmax again and remove every token whose probability is
   below `min_p` times the largest.
5. **Temperature.** Divide every logit by `temperature`.
6. **Sample.** With $\pi$ the softmax of the result and $c_k = \sum_{i \le k}
   \pi_i$, the token is the smallest $k$ with $c_k > u \cdot c_{V-1}$. `u` is a
   Python float in $[0, 1)$, supplied by the caller.

Logits may already contain `-inf` entries; they stay removed. `top_k = 1` is greedy
decoding, whatever `u` is.

**`generate(weights, config, prompt_ids, max_new_tokens, sampler, uniforms)`** is the
loop: `prefill`; then for step $i = 0, 1, \dots$: the token is
`sample_token(logits, sampler, history, uniforms[i])`, where `history` is the list
of tokens **generated so far in this call** (not the prompt). If the token is
`config["eos_token_id"]`, generation ends, and the token is neither returned nor fed
to the model. Otherwise it is appended, and if it is the `max_new_tokens`-th token
generation ends as well; else `decode_step` gives the logits of the next step.
`uniforms` is a list of `max_new_tokens` floats. The result is the list of generated
tokens as Python ints.

### Example 4

`sample_token` with `logits = [3.0, 2.0, 4.0, 0.0, 1.0, 2.5]`, `history = [2, 2, 3]`,
and `{"repetition_penalty": 2.0, "frequency_penalty": 0.0, "presence_penalty": 0.5,
"top_k": 4, "top_p": 0.85, "min_p": 0.4, "temperature": 2.0}`:

| stage | logits afterwards | note |
|---|---|---|
| penalties | `[3, 2, 1.5, -0.5, 1, 2.5]` | token 2: $4 / 2 - 0.5$; token 3: $0 \cdot 2 - 0.5$ |
| top-k 4 | `[3, 2, 1.5, -inf, -inf, 2.5]` | |
| top-p 0.85 | `[3, 2, -inf, -inf, -inf, 2.5]` | probabilities 0.455, 0.276, 0.167, 0.102 for tokens 0, 5, 1, 2: the mass before token 2 is 0.898 |
| min-p 0.4 | `[3, -inf, -inf, -inf, -inf, 2.5]` | renormalized: 0.507, 0.307, 0.186; the threshold is $0.4 \cdot 0.507$ |
| temperature 2 | `[1.5, -inf, -inf, -inf, -inf, 1.25]` | $\pi$ = 0.562 for token 0 and 0.438 for token 5 |

So `u = 0.5` gives token 0 and `u = 0.6` gives token 5. With the temperature applied
first, top-p would have kept token 2 and min-p token 1.

On the real model, the prompt of `[{"role": "user", "content": "Hi!"}]` (32 ids)
with top-k 40, top-p 0.9, min-p 0.02, temperature 0.8, and penalties 1.15, 0.2, 0.1
produces, for the ten uniforms of the fixture, nine tokens, `Hello! How can I assist
you today?`, and then end-of-sequence at the tenth draw.

## Stage 5: streaming text and stop strings

```python
def stream_chat(weights, config, tokenizer, messages, max_new_tokens, sampler, uniforms, stop): ...
```

`stream_chat` is a generator that yields the reply to a conversation as non-empty
strings. `tokenizer` is an instance of your `Tokenizer`, and `stop` a list of
non-empty strings, possibly empty.

1. **Tokens.** The prompt ids are `tokenizer.encode(chat_prompt(messages))`, and the
   reply's tokens are those of `generate` for these ids, `max_new_tokens`, `sampler`,
   and `uniforms`.
2. **Text.** After $i$ tokens, the reply's bytes are the bytes of those tokens in
   order, where a token that `tokenizer.json` flags `"special": true` contributes
   no bytes (an added token that is not special contributes its content). The
   **text** $t_i$ is those bytes decoded as UTF-8, invalid bytes as U+FFFD, except
   an unfinished character at the very end, which waits for its remaining bytes.
   This is Python's incremental decoder,
   `codecs.getincrementaldecoder("utf-8")(errors="replace")`.
3. **What may be shown.** A *match* is an occurrence of a stop string in the text.
   - If $t_i$ contains a match, take the one that **ends first** (among those, the
     one that starts first). The visible text is $t_i$ up to the start of that
     match, and **the reply is over**: no further token is generated.
   - Otherwise the visible text is $t_i$ without its longest suffix that is a
     **proper prefix** of some stop string: that suffix is held back, because it may
     become a stop string.
4. **Pieces.** After each token, yield the part of the visible text that has not
   been yielded yet, if it is not empty.
5. **The end.** When generation ends without a match (end-of-sequence, or
   `max_new_tokens` tokens), an unfinished character becomes one U+FFFD, and the
   rest of the text is released: if it now contains a match, it is cut at the match
   as in step 3; otherwise all of it is visible, including what was held back.
   Yield what has not been yielded, if it is not empty.

So the concatenation of the pieces is the reply up to the first stop string, and the
pieces themselves are determined: one per token at most, plus one at the end.

### Example 5

The tiny model's reply to `[{"role": "user", "content": "Hi!"}]` with `stop =
["ro\r"]` (`\r` is a carriage return), top-k 12, temperature 1.6, repetition penalty
1.3, and `max_new_tokens = 20`:

| token | its bytes | yielded | why |
|---|---|---|---|
| 1 | `or` | `"o"` | `r` could begin the stop string: held |
| 2 | `96` (hex) | `"r\ufffd"` | a stray continuation byte; the `r` was not a stop |
| 3 | `16` (hex) | `"\x16"` | |
| 4 | `(` | `"("` | |
| 5 | `st` | `"st"` | |
| 6 | `F0` (hex) | nothing | the first byte of a four-byte character |
| 7 | `14` (hex) | `"\ufffd\x14"` | the character was never finished |
| 8 | `ser` | `"se"` | `r` is held again |
| 9 | `en` | `"ren"` | |
| 10 | `E` | `"E"` | |
| 11 | ` pro` | `" p"` | `ro` is held |
| 12 | a carriage return | nothing | `ro\r` is complete: the reply is over |

Twelve of the twenty tokens are generated. Without stop strings, the same uniforms
give twenty tokens; the seventeenth is the special token `<|unk|>`, which is fed to
the model and yields nothing.

On the real model, `system: "Be brief."`, `user: "Count from one to ten in words."`
with `top_k = 1` begins `One, two, three, four, ...`. With `stop = ["\n\n", ",
three"]` the pieces are `"One"` and `", two"`: each comma is held until the next
token shows that ` three` does not follow, and the fifth token completes the stop
string.

## Shapes

| name | shape | dtype |
|---|---|---|
| values of `load_weights` | as in the file's header | float32 |
| `ids`, `prompt_ids` | `(T,)`, `(P,)` | integer |
| result of `forward` | `(T, V)` | float32 |
| logits of `prefill`, `decode_step`; argument of `sample_token` | `(V,)` | float32 |
| `cache["keys"][i]`, `cache["values"][i]` | `(num_key_value_heads, C, head_dim)`, $C \ge$ `cache["length"]` | float32 |
| `model.embed_tokens.weight` | `(V, d)` | |
| `self_attn.q_proj.weight`, `o_proj.weight` | `(d, d)` | |
| `self_attn.k_proj.weight`, `v_proj.weight` | `(H_kv * head_dim, d)` | |
| `shared_mlp.input_linear.weight` | `(2 * d_ff, d)` | |
| `shared_mlp.output_linear.weight` | `(d, d_ff)` | |
| `input_layernorm.weight`, `post_attention_layernorm.weight`, `model.norm.weight` | `(d,)` | |

## Constraints

- **Oracle.** Expected values come from the kit's own implementations of the same
  definitions (a numpy forward pass with its own cache, and its tokenizer), stored
  in the problem's fixtures; they were checked against `transformers` (logits to
  7e-5 on the real model) and `tokenizers` when the fixtures were made.
- **Exact:** weights (bfloat16 and float16 convert to float32 without rounding),
  token ids, prompt texts, decoded strings, sampled tokens, and the list of streamed
  pieces. `bytes_read` has the limit of stage 1, with no tolerance.
- **Logits, keys, and values:** a relative tolerance of `1e-3` and an absolute
  tolerance of `1e-3`; on the real model at 64 fixed vocabulary columns, and for
  three layers of the cache. The earlier real-model problems measured float32
  implementations in numpy, PyTorch, JAX, and MLX to agree to about `1e-4` on logits
  of magnitude 25, so the tolerance has a factor of ten to spare.
- **Sampling is exact** because every uniform number in the tests was chosen so that
  its token survives a perturbation of the logits by `3e-3` and of the uniform by
  `1e-4`. Float noise cannot change a token; a different order of the stages can.
- Prompts have at most 65 tokens and at most 20 tokens are generated. The real
  weights are loaded once per run, by your `load_weights`, and reused by the later
  stages; so is your `Tokenizer`.
- The time limit is 30 seconds per case. The reference needs about 3 seconds for all
  cases in PyTorch and MLX and about 8 in JAX (about 15 the first time, while JAX
  compiles).

> [!NOTE]
> JAX compiles every operation once per combination of input shapes. A cache that
> grows by one slot per token changes the shapes of the attention at every step.
> The capacity $C$ in the cache layout exists for this: keep buffers whose capacity
> grows in larger steps, mask the unused slots, and write new arrays with `.at[]`.
> In PyTorch and MLX, concatenating is fine, and $C$ can equal the length.

## Why this matters

This is what `llama.cpp`, `mlx-lm`, and the model class behind `transformers`'
`generate` are, with the engineering for speed taken out: a file format, a
tokenizer, a template, a forward pass, a cache, a sampler, a detokenizer. Each
piece is small, and each fails silently. Offsets counted from the start of the file
give tensors of the right shape filled with JSON. A loader that reads everything to
return one layer turns a 5 ms inspection into a 705 MB read. A merge that crosses a
word, or ChatML sent to a model trained on another layout, still produces fluent
text, a little worse. The textbook attention scale, a residual multiplier applied to
one branch, the fused MLP matrix split the wrong way round: all produce logits, and
only a reference tells you they are the wrong ones. A key rotated for position 0
degrades long replies only. Temperature applied before the filters changes which
tokens survive. And a stop string that was already shown to the user cannot be
taken back. Having built the whole path once, you can read any runtime's source and
know which of these it is doing on a given line.
