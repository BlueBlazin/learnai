# Whetstone curriculum

The complete catalog plan: every problem Whetstone will ship, in fifteen tracks, with the
milestone in which each is authored. Authors write problems from this document, five at
a time, without further coordination; if a spec here is ambiguous, fix the spec here
first, then write the problem.

Read with `docs/ARCHITECTURE.md` section 3 (format, judges) and
`docs/PROBLEM_AUTHORING.md` (test API, kit, house style). Facts about models and
engines were checked on 2026-11-14; see [Sources](#sources).

## 1. How to read this document

Each track has a table (one row per problem) followed by one short spec per problem.

| column | meaning |
|---|---|
| order | `order` in `problem.toml`. Multiples of 10; existing problems keep their slot. |
| id | Globally unique, kebab-case; equals the directory name. |
| diff | `easy`, `medium`, `hard`. |
| judges | Every judge type the tests use. |
| fw | `t/j/m` = `["torch", "jax", "mlx"]`; `python` = `["python"]`; `mlx` = `["mlx"]`; `torch` = `["torch"]`. |
| prereq | `prerequisites` in `problem.toml` (ids). `-` means none. |
| ms | Milestone: `M1` (exists today), `M2`, `M3`, `M4`, `M5`. |

Rules that apply to every spec, so they are not repeated:

- **Self-contained.** A problem never imports the learner's earlier solutions. Building
  blocks from earlier problems come from the kit (section 3). Prerequisites are advice
  to the learner, not a code dependency.
- **Functional.** State in, state out; randomness is an argument (`u`, `uniforms`).
  Where a spec says "class", the tensor-framework version is a pair of pure functions
  over a state tuple unless the problem is `python`, `mlx`, or `torch` only.
- **`python` problems** may import the standard library and numpy unless the spec
  forbids it.
- **Named sources** (author and year, or system) are the primary source the author must
  verify the specification against. Where an engine and a paper disagree, the spec
  says which one the problem follows.
- **Mutants.** "Mutants:" lists the classic mistakes; each becomes one mutant with a
  case written for it. Authors add the forbidden-shortcut mutant where a `forbidden`
  list applies.
- **[kit: X]** marks a kit object that does not exist yet (section 3).
- **requires** is `cpu` unless the track says otherwise.

### Catalog totals

| # | track | title | total | M1 | M2 | M3 | M4 | M5 | easy | medium | hard |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `foundations` | Tensor Foundations | 13 | 2 | 6 | 2 | 0 | 3 | 10 | 3 | 0 |
| 2 | `tokenization` | Tokenization | 13 | 1 | 5 | 2 | 0 | 5 | 5 | 6 | 2 |
| 3 | `transformer` | Transformer Forward Pass | 19 | 2 | 9 | 2 | 0 | 6 | 8 | 8 | 3 |
| 4 | `sampling` | Sampling and Decoding | 16 | 2 | 8 | 0 | 0 | 6 | 6 | 8 | 2 |
| 5 | `kv-cache` | KV Cache and Attention Memory | 16 | 1 | 8 | 1 | 0 | 6 | 4 | 10 | 2 |
| 6 | `constrained` | Constrained Generation | 13 | 0 | 6 | 0 | 0 | 7 | 2 | 8 | 3 |
| 7 | `quantization` | Quantization and Compression | 18 | 1 | 7 | 2 | 0 | 8 | 5 | 11 | 2 |
| 8 | `speculative` | Speculative and Accelerated Decoding | 14 | 1 | 5 | 0 | 0 | 8 | 4 | 5 | 5 |
| 9 | `sizing` | Sizing and Benchmarking | 11 | 0 | 6 | 1 | 0 | 4 | 9 | 2 | 0 |
| 10 | `mlx` | MLX GPU Programming | 18 | 0 | 0 | 18 | 0 | 0 | 6 | 8 | 4 |
| 11 | `cuda` | CUDA GPU Programming | 14 | 0 | 0 | 0 | 14 | 0 | 5 | 6 | 3 |
| 12 | `serving` | Serving Systems | 22 | 2 | 9 | 0 | 0 | 11 | 5 | 14 | 3 |
| 13 | `systems` | Distributed and LLM Application Systems | 12 | 0 | 0 | 0 | 0 | 12 | 4 | 6 | 2 |
| 14 | `frontier` | Frontier | 8 | 0 | 0 | 0 | 0 | 8 | 0 | 4 | 4 |
| 15 | `capstones` | Capstones | 6 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 6 |
| | **all** | | **213** | **12** | **69** | **28** | **14** | **90** | **73** | **99** | **41** |

Difficulty split: 34% easy, 46% medium, 19% hard. At the end of M2 the catalog has 81 problems. M3 is the 18 MLX problems plus 10 real-model problems in other tracks (two more real-model problems are inside the MLX track).

## 2. Real models for milestone 3

GPT-2 and SmolLM2-class models are ruled out as outdated. The two models below are the
defaults for every "real model" problem. Both are ungated, Apache-2.0, plain
decoder-only transformers whose repositories contain no custom modeling code, and they
share one byte-level BPE `tokenizer.json`, so the tokenizer problems cover both.

| role | repo id | params | files | license |
|---|---|---|---|---|
| small (fast tests) | [`ibm-granite/granite-4.0-350m`](https://huggingface.co/ibm-granite/granite-4.0-350m) | 352.4 M, BF16 | `model.safetensors`, 704.8 MB | Apache-2.0, ungated |
| large | [`ibm-granite/granite-4.2-3b`](https://huggingface.co/ibm-granite/granite-4.2-3b) (released 2026-08-25) | 3.66 B, BF16 | two shards, 5.36 + 1.96 GB | Apache-2.0, ungated |
| large, 8-bit (what problems load) | [`lmstudio-community/granite-4.2-3b-MLX-8bit`](https://huggingface.co/lmstudio-community/granite-4.2-3b-MLX-8bit) | 8-bit affine, group size 64 | `model.safetensors`, 3.89 GB | Apache-2.0 |
| small, 8-bit (optional) | [`mlx-community/granite-4.0-350m-8bit`](https://huggingface.co/mlx-community/granite-4.0-350m-8bit) | 8-bit | 374.5 MB | Apache-2.0 |

Architecture facts read from `config.json`, `tokenizer.json`, and the safetensors
headers (append `/raw/main/config.json` or `/tree/main` to a repo URL):

| fact | granite-4.0-350m | granite-4.2-3b |
|---|---|---|
| `model_type` | `granitemoehybrid` (all layers are attention layers, zero experts) | `granite` |
| layers / hidden size | 28 / 1024 | 40 / 2560 |
| attention | grouped-query: 16 heads, 4 KV heads, head dim 64 | grouped-query: 40 heads, 8 KV heads, head dim 64 |
| attention scale | `1/64` (that is `1/head_dim`, not `1/sqrt(head_dim)`) | `1/64` |
| MLP | SwiGLU, intermediate 2048; gate and up projections fused in `shared_mlp.input_linear`, shape `[4096, 1024]` | SwiGLU, intermediate 8192; separate gate, up, down |
| norm | RMSNorm, pre-norm, eps 1e-5 | RMSNorm, pre-norm, eps 1e-5 |
| positions | RoPE, theta 1e7 | RoPE, theta 1e7 |
| multipliers | embedding x12, logits /4, residual x0.263 | all 1.0 |
| embeddings | tied | **not tied** |
| vocabulary / context | 100352 / 32768 | 100352 / 131072 |
| tokenizer | byte-level BPE, `tokenizer.json` | same |
| chat template | ChatML-style | ChatML-style with `<think>`; `enable_thinking` defaults to true |

Not verified (the first M3 author must check before writing fixtures): that
`transformers` loads both model types natively (inferred from the absence of `auto_map`
and modeling `.py` files); pre-norm placement (inferred from tensor names
`input_layernorm`, `post_attention_layernorm`); the exact place each multiplier is
applied. The multipliers are a feature for teaching: "match the reference logits" fails
loudly until every one is in the right place.

**Fallback pair** if Granite turns out to need custom code:
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) (751.6 M stored, 1.50 GB;
28 layers, hidden 1024, 16 heads / 8 KV heads, head dim 128, SwiGLU 3072, RMSNorm with
QK-norm, RoPE theta 1e6, tied, vocabulary 151936, BPE) and
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
(4.02 B; 36 layers, hidden 2560, 32 / 8 heads, SwiGLU 9728, RoPE theta 5e6) with
[`mlx-community/Qwen3-4B-Instruct-2507-8bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-8bit)
(4.27 GB). Cleaner arithmetic, but from 2025, and the small one is not "well under 1B".

Rejected: Qwen3.5 small models (multimodal, three of four layers are linear attention),
Gemma 4 E2B/E4B (any-to-any, 5.1 B total, no sub-1B variant), LFM2.5 and LFM3
(non-permissive license, convolution hybrid, custom code), `google/gemma-3-270m`
(gated).

**Rules for real-model problems.** Weights are never committed. The kit downloads them
on demand into `.whetstone/assets/` [kit: `assets`], and a problem whose assets are
absent reports every case as `skipped` with a message naming the download command.
Reference outputs (logits, token ids, text) are small fixture files committed inside
the problem directory, generated once by a script the author also commits
(`fixtures/make.py`), using `transformers`, `tokenizers`, or `mlx-lm` as the reference
implementation. Tests use prompts of at most 16 tokens on the small model so a CPU
forward pass stays under two seconds. Problems on the 3B model always load the 8-bit
file and are `mlx`-only.

## 3. Kit additions needed

Proposed names and one-line interfaces, grouped by the milestone that first needs them.
All follow the kit rules in `PROBLEM_AUTHORING.md` section 7 (seeded, numpy or plain
Python inside, the solution's framework at the boundary, functional, `counters`,
`with_framework`).

### Needed by M2

| name | interface | first used by |
|---|---|---|
| `KVCache.truncate` | `cache.truncate(n) -> KVCache` keeps the first `n` tokens; `cache.select(rows) -> KVCache` reorders or repeats batch rows. | `prefix-cache-reuse`, `beam-search`, `speculative-decode-loop` |
| `draft_target_pair` | `draft_target_pair(fw, seed, agreement=0.8) -> (draft, target)`: two `TinyLM`s over one vocabulary; the draft is the target with seeded weight noise, tuned so greedy tokens agree at about the given rate. Separate counters. | `speculative-decode-loop` |
| `ops` | `whetstone.kit.ops`: framework-native reference building blocks, one signature in all frameworks: `softmax`, `rms_norm`, `rope`, `attention(q, k, v, mask)`, `swiglu`, `top_k_filter`, `top_p_filter`, `min_p_filter`. Lets a later problem use an earlier concept without re-deriving it. | `pre-norm-block`, `sampler-chain` |
| `DecoderConfig`, `decoder_params` | `decoder_params(cfg, seed) -> dict[str, ndarray]` for a modern block layout (RMSNorm pre-norm, RoPE, GQA via `n_kv_heads`, SwiGLU, tied head), named like Hugging Face checkpoints; `decoder_forward_ref(cfg, params, tokens)` numpy oracle. | `decoder-forward`, `padded-batch-invariance` |
| `ToyVocab` | `ToyVocab(size=64, seed)`: token id to byte string table aligned with `TinyLM` ids (single bytes, digits, JSON punctuation, multi-character and multi-byte UTF-8 pieces, named special tokens), plus a small ranked merge table. `vocab.bytes(id)`, `vocab.specials`. | `streaming-detokenizer`, `special-tokens-split`, `dfa-token-mask`, `stop-sequences` |
| `CharDFA` | `CharDFA(transitions, start, accepting)`: `dfa.step(state, byte) -> state or DEAD`, `dfa.counters.steps`; `CharDFA.from_regex(pattern)` for tests. | `dfa-token-mask`, `dfa-token-index` |
| `TokenConstraint` | Protocol used by decoding loops: `start() -> state`, `allowed(state) -> list[int]`, `advance(state, token) -> state`, `is_accepting(state) -> bool`; reference `DFAConstraint(dfa, vocab)`. | `constrained-greedy-decode` |
| `SimEngine` | `SimEngine(sim, cost=CostModel(prefill_s_per_token, decode_base_s, decode_s_per_seq), max_batch_tokens=, kv_blocks=None)`: `await engine.step(prefill=[(seq_id, n_tokens)], decode=[seq_id, ...])` takes virtual time by the cost model, advances each sequence, records `step`, `first_token`, `token`, `finish` trace events, and raises on protocol violations (budget exceeded, decoding a sequence whose prefill is incomplete, memory exhausted). | `continuous-batching-scheduler`, `chunked-prefill-scheduling`, `preemption-recompute` |
| `SimRequest`, `workloads` | `SimRequest(id, arrival_s, prompt_tokens, output_tokens, priority=0, tenant=None, deadline_s=None)`; `workloads.poisson(rng, rate, n, ...)`, `workloads.bursty(...)`, `workloads.replay(list)`. | all serving simclock problems |
| `ScriptedLM` | `ScriptedLM(script)`: async token source for service problems, `async for piece in lm.stream(prompt, max_tokens)`, with a test-controlled gate and a `closed` flag. | `openai-chat-endpoint`, `metrics-endpoint` |

### Needed by M3

| name | interface | first used by |
|---|---|---|
| `assets` | `assets.path("granite-4.0-350m", "model.safetensors") -> Path` or raises `AssetMissing`; registry of repo id, revision hash, file list, sizes, SHA-256; CLI `ws assets pull <name>`. `problem.toml` gains `assets = ["granite-4.0-350m"]`; the harness skips all cases when an asset is missing. | every real-model problem |
| `fixtures` | `ctx.fixture("logits.npz")` loads a file from the problem's `fixtures/` directory. | every real-model problem |
| MLX-only oracle | `ws validate` accepts `frameworks = ["mlx"]` with `reference/mlx.py` as the oracle (today a tensor problem must list `torch`). | track `mlx` |
| `EvalSpy` | `with EvalSpy() as spy:` counts `mx.eval`, `mx.async_eval`, and implicit evaluations (`.item()`, `.tolist()`, `np.array(x)`, printing) with the call site; `spy.evals`, `spy.implicit`. | `mlx-lazy-eval-placement`, `mlx-streaming-generate` |
| `TraceCounter` | `tc = TraceCounter(); y = tc.mark(x)` is an identity whose Python body increments `tc.count`, so the count is the number of times a compiled function was traced. | `mlx-compile-step`, `mlx-compile-shapes` |
| `MLXTinyLM` | A `TinyLM`-shaped model written in `mlx.nn` (same call convention, plus `.parameters()`), traceable by `mx.compile`, seeded, with `layers`, `n_kv_heads`, `rope` options. | `mlx-compile-step`, `mlx-batched-generate` |
| `CountingFile` | File-like wrapper counting `bytes_read` and distinct pages touched; works with `mmap`. | `mlx-quantized-checkpoint-load` |
| `mlx_memory` | `with mlx_memory() as m:` resets and reads `mx.get_peak_memory()` and `mx.get_active_memory()`; `m.peak_bytes`. | `mlx-memory-accounting`, `mlx-dtype-discipline` |

### Needed by M4

| name | interface | first used by |
|---|---|---|
| per-case requirement | `@case(..., requires="cuda")`: the case is `skipped` when CUDA is absent; all other cases of a `requires = "cuda"` problem still run on CPU. Without this, the CUDA track cannot be validated on the owner's machine. | track `cuda` |
| `SyncSpy` | `with SyncSpy() as spy:` counts host synchronizations made from the solution file (`Tensor.item`, `.tolist`, `.cpu`, `.numpy`, `bool(tensor)`, `int(tensor)`, `print`); works on CPU tensors. | `cuda-sync-free-decode-loop` |
| `FakeDevice` | `FakeDevice()`: `dev.sync()` and `dev.clock()` stand in for `torch.cuda.synchronize` and `time.perf_counter`; lazily "finishes" work only on `sync`, so a timer read without a sync measures zero. | `cuda-timing-sync` |
| `OOMModel` | `OOMModel(fn, max_items)`: callable that raises `torch.OutOfMemoryError` when given a batch larger than `max_items`; counts calls. | `cuda-oom-backoff` |
| `FakeCudaGraph` | Same surface as `torch.cuda.CUDAGraph` (`capture_begin`, `capture_end`, `replay`): records the `data_ptr()` of tensors seen during capture and recomputes from those storages on replay, so rebinding a placeholder instead of copying into it gives stale output on CPU exactly as on GPU. | `cuda-graph-replay` |
| `tritonshim` | `from whetstone.kit import tritonshim as tl`: a numpy implementation of the subset of `triton.language` the track uses (`program_id`, `arange`, `load`, `store` with masks, `where`, `sum`, `max`, `sqrt`, `exp`), plus `launch(kernel, grid, **args)`. On a CUDA machine with Triton installed the same kernel source is compiled by real Triton. | `cuda-triton-rmsnorm` |

### Needed by M5

| name | interface | first used by |
|---|---|---|
| `LayeredLM` | A `TinyLM` that exposes layers: `h = lm.embed(tokens)`, `h, cache = lm.layers(h, start, stop, cache)`, `lm.head(h)`; `counters.layer_calls`, `counters.layer_tokens`. | `early-exit-self-speculation`, `cross-layer-kv-sharing` |
| `TinyMoE`, `CountingExperts` | `experts(i, x)` runs expert `i` on rows `x` and counts `expert_calls` and `expert_rows`; `TinyMoE` is a decoder with MoE MLPs. | `moe-sparse-dispatch`, `expert-parallel-dispatch` |
| `SimDevices` | `SimDevices(memory_bytes=[...])`: `devs.put(i, name, array)`, `devs.send(src, dst, array)`; counts bytes placed and `comm_bytes`, raises over budget. | `tensor-parallel-linear`, `expert-parallel-dispatch` |
| `Grammar` | Parsed GBNF subset: `Grammar.parse(text)`, `g.rules`, and an oracle `g.accepts(text)`, `g.accepts_prefix(text)`. | `grammar-token-mask` |
| `ScriptedChatModel` | `model.complete(messages) -> message` replaying a script keyed on the conversation; counts calls; can emit tool calls, malformed arguments, and parallel calls. | `tool-call-loop`, `cascade-router`, `speculative-tool-calls` |
| `TinyEncoder`, `CountingScorer` | Seeded bidirectional encoder returning token states; cross-encoder scorer `scorer(batch_of_pairs)` counting `padded_tokens`. | `embedding-pooling`, `rerank-batching` |
| `RecurrentLM` | Model whose per-layer state is a fixed-size tensor that cannot be truncated; `lm(tokens, state) -> (logits, state)`; counters. | `hybrid-state-checkpoints` |
| `SimCluster` | Replicas of `SimEngine` with per-replica prefix caches, startup delay, and a `route` hook. | `prefix-aware-router`, `autoscaler-policy`, capstones |
| stage grouping | `@case(..., stage="2: paged KV cache")`: the UI groups cases and shows per-stage progress. Capstones work without it (stage prefix in the case name) but read better with it. | track `capstones` |

## 4. The catalog

### Track 1: `foundations` Tensor Foundations (order 10)

The numerics and tensor mechanics every later track leans on.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `stable-softmax` | Numerically Stable Softmax | easy | numeric | t/j/m | - | M1 |
| 20 | `rmsnorm` | RMSNorm | easy | numeric, property | t/j/m | - | M1 |
| 30 | `gelu-silu-activations` | GELU and SiLU | easy | numeric | t/j/m | - | M2 |
| 40 | `split-merge-heads` | Splitting and Merging Attention Heads | easy | numeric, property | t/j/m | - | M2 |
| 50 | `attention-mask-builder` | Causal and Padding Masks | easy | numeric | t/j/m | - | M2 |
| 60 | `last-token-gather` | Gathering the Last Real Token | easy | numeric | t/j/m | - | M2 |
| 70 | `sequence-log-likelihood` | Sequence Log-Likelihood and Perplexity | easy | numeric | t/j/m | `stable-softmax` | M2 |
| 80 | `bf16-decode` | Decoding bfloat16 by Hand | medium | numeric, property | t/j/m | - | M2 |
| 90 | `layernorm` | LayerNorm | easy | numeric, property | t/j/m | `rmsnorm` | M5 |
| 100 | `kl-divergence-from-logits` | KL Divergence Between Logit Sets | easy | numeric, property | t/j/m | `stable-softmax` | M5 |
| 110 | `half-precision-rmsnorm` | RMSNorm That Survives float16 | medium | numeric | t/j/m | `rmsnorm` | M5 |
| 120 | `safetensors-parse` | Parsing safetensors by Hand | medium | numeric | python | `bf16-decode` | M3 |
| 130 | `real-model-perplexity` | Perplexity of a Real Model | easy | numeric | t/j/m | `sequence-log-likelihood` | M3 |

**`gelu-silu-activations`** — Implement `gelu(x, approximate)` (exact erf form and the tanh approximation of Hendrycks and Gimpel, 2016) and `silu(x)`. Goal: know which activation a checkpoint means and keep both stable at large magnitude. Numeric against the oracle, including `|x|` up to 1e4. Mutants: tanh form returned for `approximate="none"`; `x * sigmoid(x)` computed as `x / (1 + exp(-x))` overflowing to NaN at `-inf`-like inputs; wrong constant `sqrt(2/pi)`. Forbid the framework built-ins.

**`split-merge-heads`** — Implement `split_heads(x, n_heads)` mapping `(B, T, D)` to `(B, H, T, D/H)` and its inverse `merge_heads`. Goal: reshape and transpose are different operations. Numeric plus the round-trip property. Mutants: reshape without transpose (passes when `T == H`, so tests avoid and then include that shape); merge that transposes the wrong pair; contiguous-only code that breaks on a transposed input.

**`attention-mask-builder`** — Implement `build_mask(lengths, T, padding_side)` returning an additive float32 mask of shape `(B, 1, T, T)` that combines the causal rule with key padding for left- or right-padded batches. Rows of padding queries must not become all `-inf` (specified: a padding query attends to itself). Mutants: mask on queries instead of keys; `-1e9` in place of `-inf`; all-masked rows that make softmax NaN; left and right padding swapped; off-by-one on the diagonal.

**`last-token-gather`** — Implement `last_token_logits(logits, lengths, padding_side)` selecting, for each row of a padded `(B, T, V)` batch, the logits at the last real token. Goal: the indexing bug behind "batched generation gives garbage for short prompts". Numeric. Mutants: always `[:, -1]`; `lengths` used as the index (off by one); gather along the wrong axis.

**`sequence-log-likelihood`** — Implement `sequence_logprob(logits, tokens, mask)` returning per-row sum of log-probabilities of `tokens[t+1]` under `logits[t]` over unmasked positions, and `perplexity(...)`. Goal: the shift-by-one alignment used by every evaluation and by speculative verification. Numeric, with extreme logits. Mutants: no shift; averaging over padded positions; `log(softmax)` underflow; perplexity as `exp` of the sum instead of the mean.

**`bf16-decode`** — Implement `bf16_to_f32(bits)` (uint16 array to float32) and `f32_to_bf16(x)` with round-to-nearest-even, using integer operations only. Goal: bfloat16 is the top 16 bits of float32, and what truncation versus rounding costs. Numeric and exact-bit round-trip property, including NaN, infinities, subnormals, and ties. Mutants: truncation instead of rounding; rounding that overflows the mantissa into the wrong exponent; treating the bits as IEEE half precision.

**`layernorm`** — Implement `layer_norm(x, weight, bias, eps)` over the last axis. Goal: contrast with RMSNorm (mean subtraction, biased variance, eps inside the square root). Numeric plus shift-invariance property. Mutants: unbiased variance; eps outside the root; no mean subtraction.

**`kl-divergence-from-logits`** — Implement `kl_from_logits(p_logits, q_logits)` per row, computed from log-softmax, defined as 0 where `p` is 0 and `+inf` where `q` is 0 and `p` is not. Goal: the metric used to judge quantization and speculative drafts. Numeric with masked entries; non-negativity property. Mutants: arguments swapped; `0 * -inf` producing NaN; computed from probabilities with underflow.

**`half-precision-rmsnorm`** — Implement `rms_norm_f16(x, weight, eps)` for float16 input and output without upcasting, correct for `|x|` up to 6e4 where `x * x` overflows float16. Goal: where half precision breaks and the rescale-first fix. Numeric against a float32 oracle rounded to float16 (tolerance stated for float16); result dtype must be float16. Mutants: naive square (inf, then 0 output); upcast to float32 (forbidden by a dtype spy case); rescale by the mean instead of the max.

**`safetensors-parse`** — Implement `read_header(buf)` and `read_tensor(buf, name)` for the safetensors format: 8-byte little-endian header length, JSON header with `dtype`, `shape`, `data_offsets` relative to the end of the header, `__metadata__`; return numpy arrays without copying (BF16 as uint16). Judged on synthetic files and on `granite-4.0-350m` (tensor count, shapes, checksums from fixtures). Mutants: offsets taken from file start; big-endian length; `__metadata__` treated as a tensor; a copy where a view is required. Forbid the `safetensors` package. [kit: `assets`]

**`real-model-perplexity`** — Implement `perplexity(forward, token_ids)` where `forward` is a kit-provided callable running `granite-4.0-350m` and returning logits; compute the perplexity of a fixture passage in windows of at most 16 tokens with the specified stride. Goal: the first contact with real logits, and the windowing arithmetic. Numeric against a fixture value. Mutants: double-counting overlapped tokens; scoring the first token of each window with no context; natural log versus base-2 confusion. [kit: `assets`]

### Track 2: `tokenization` Tokenization (order 20)

Text in, ids out, and back, exactly as production tokenizers do it.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `bpe-encode` | Byte-Level BPE Encoding | medium | numeric, property | python | - | M1 |
| 20 | `utf8-boundaries` | UTF-8 Character Boundaries | easy | numeric, property | python | - | M2 |
| 30 | `pad-batch-left` | Padding a Batch for Generation | easy | numeric | python | - | M2 |
| 40 | `special-tokens-split` | Special Tokens Without Injection | medium | numeric | python | `bpe-encode` | M2 |
| 50 | `streaming-detokenizer` | Streaming Detokenization | medium | numeric, property, cost | python | `utf8-boundaries` | M2 |
| 60 | `chat-template-render` | Rendering a Chat Template | easy | numeric | python | `special-tokens-split` | M2 |
| 70 | `wordpiece-encode` | WordPiece Encoding | easy | numeric | python | - | M5 |
| 80 | `context-window-trim` | Fitting a Conversation into the Context Window | easy | numeric, property | python | `chat-template-render` | M5 |
| 90 | `bpe-learn-merges` | Learning a BPE Merge Table | medium | numeric | python | `bpe-encode` | M5 |
| 100 | `bpe-encode-fast` | BPE Encoding in O(n log n) | hard | numeric, speed | python | `bpe-encode` | M5 |
| 110 | `unigram-viterbi` | Unigram Tokenization with Viterbi | hard | numeric, property | python | - | M5 |
| 120 | `tokenizer-json-load` | Running a Real Tokenizer's Merges | medium | numeric | python | `bpe-encode`, `special-tokens-split` | M3 |
| 130 | `real-model-chat-turn` | One Chat Turn with a Real Model | medium | numeric | python | `chat-template-render`, `streaming-detokenizer` | M3 |

**`utf8-boundaries`** — Implement `complete_prefix_len(data: bytes) -> int`: the length of the longest prefix that ends on a character boundary, treating invalid bytes as complete (they will decode to U+FFFD), and holding back only a valid-so-far incomplete trailing sequence. Goal: the primitive under streaming detokenization. Numeric plus a property over random splits of valid text. Mutants: holds back after any byte `>= 0x80`; ignores 4-byte sequences; holds back forever on an invalid lead byte.

**`pad-batch-left`** — Implement `pad_batch(seqs, pad_id, side)` returning `(ids, attention_mask, lengths)` as nested lists, left-padding by default. Goal: decoder-only generation needs the last column to be real tokens. Numeric. Mutants: right padding for `side="left"`; mask inverted; empty sequence crash (specified: a row of pads with an all-zero mask is an error, `ValueError`).

**`special-tokens-split`** — Implement `encode_with_specials(text, specials, encode_ordinary, allowed)`: split `text` on special-token strings by longest match at the earliest position, map them to their ids only when listed in `allowed`, and raise `ValueError` if a disallowed special string appears (the tiktoken rule), with `allowed="none_as_text"` encoding them as ordinary text. Goal: prompt injection through `<|im_start|>` in user text. Numeric. Mutants: shortest match; specials always parsed; overlapping specials resolved right to left. [kit: `ToyVocab`]

**`streaming-detokenizer`** — Implement `detok_step(state, token_id, vocab) -> (state, text)` and `detok_flush(state) -> text`: emit text incrementally such that the concatenation equals the full decode, never emitting a partial UTF-8 character or a premature U+FFFD. Cost: total bytes decoded is linear in the output, counted by `vocab.counters.bytes_decoded` (re-decoding the whole sequence each step is quadratic). Mutants: per-token decode (splits multi-byte characters); full re-decode each step; never flushing an invalid tail. [kit: `ToyVocab`]

**`chat-template-render`** — Implement `render_chat(messages, add_generation_prompt, bos)` for the ChatML layout (`<|im_start|>role\ncontent<|im_end|>\n`), with a default system message when none is given, `tool` role messages, and the assistant prefix when `add_generation_prompt` is true; return the string and the list of segments marked special or text so content is never parsed for specials. Numeric against literal expected strings. Mutants: missing trailing newline; generation prompt always added; BOS doubled; user content parsed for specials.

**`wordpiece-encode`** — Implement `wordpiece(word, vocab, unk, max_chars)`: greedy longest-match-first with the `##` continuation prefix, whole word to `unk` if any piece fails (Devlin et al., 2018; used by current embedding and reranking models). Numeric. Mutants: shortest match; `unk` for the failing piece only; continuation prefix on the first piece.

**`context-window-trim`** — Implement `trim_messages(messages, count_tokens, max_context, max_new_tokens)`: keep the system message and the latest user message, then drop oldest turns in whole user-assistant pairs until the prompt plus reserved output fits; raise `ValueError` if the mandatory part alone does not fit. Numeric plus the property "result fits and is a subsequence". Mutants: no reservation for output; drops the system message; orphans an assistant message.

**`bpe-learn-merges`** — Implement `learn_merges(corpus_words, n_merges)` over a word-frequency dict: repeatedly merge the most frequent adjacent pair, ties broken by the lexicographically smallest pair (Sennrich et al., 2016). This builds a tokenizer; it is not model training. Numeric against the oracle's merge list. Mutants: ties by first occurrence; counts not weighted by word frequency; overlapping pairs in `aaa` double counted.

**`bpe-encode-fast`** — Same contract as `bpe-encode`, on inputs of 50,000 bytes with a single long word: implement the linked-list plus priority-queue algorithm so encoding is O(n log n). Speed: within 5x of the reference, where the quadratic rescanning solution is more than 50x slower. Mutants: the M1 quadratic reference (correct but slow); stale heap entries applied; merge order wrong for overlapping pairs.

**`unigram-viterbi`** — Implement `unigram_encode(text, pieces)` where `pieces` maps strings to log-probabilities: return the segmentation maximizing the total score by dynamic programming over character positions, with an unknown-character fallback at a stated penalty and ties broken toward fewer, then earlier-longer, pieces (Kudo, 2018; SentencePiece). Property: no segmentation scores higher (checked by brute force on short inputs). Mutants: greedy longest match; the fallback penalty omitted; byte positions used on multi-byte text.

**`tokenizer-json-load`** — Implement `load_tokenizer(tokenizer_json: dict) -> encode, decode` for the Granite `tokenizer.json`: read the vocabulary, the ranked merges, the added and special tokens, and the byte-level pre-tokenizer, then encode fixture strings to exactly the ids the `tokenizers` library produces. Numeric on fixtures covering whitespace runs, emoji, CJK, and special tokens. Mutants: merges ranked by vocabulary id; missing leading-space handling; added tokens matched after BPE. Forbid `tokenizers`. [kit: `assets`]

**`real-model-chat-turn`** — Implement `chat_turn(messages, tokenizer, step, max_new_tokens)` where `tokenizer` and the single-token `step` function for `granite-4.0-350m` come from the kit: render the Granite chat template by hand, generate greedily to the end-of-turn token, and yield decoded text pieces. Judged on exact prompt ids and exact yielded text against fixtures. Mutants: generation prompt missing; stop token emitted as text; multi-byte characters split across yields. [kit: `assets`]

### Track 3: `transformer` Transformer Forward Pass (order 30)

Every piece of a current decoder-only model's forward pass, then the whole thing on real weights.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `causal-self-attention` | Causal Self-Attention | medium | numeric, property | t/j/m | - | M1 |
| 20 | `rope` | Rotary Position Embeddings (RoPE) | medium | numeric, property | t/j/m | - | M1 |
| 30 | `swiglu-mlp` | SwiGLU Feed-Forward | easy | numeric | t/j/m | `gelu-silu-activations` | M2 |
| 40 | `grouped-query-attention` | Grouped-Query Attention | easy | numeric, property | t/j/m | `causal-self-attention` | M2 |
| 50 | `pre-norm-block` | A Pre-Norm Decoder Block | easy | numeric | t/j/m | `causal-self-attention`, `rmsnorm` | M2 |
| 60 | `decoder-forward` | Full Decoder Forward Pass | medium | numeric, property | t/j/m | `pre-norm-block`, `rope` | M2 |
| 70 | `last-position-head` | Unembedding Only What You Need | easy | numeric, speed | t/j/m | - | M2 |
| 80 | `sliding-window-attention` | Sliding-Window Attention | easy | numeric, property | t/j/m | `causal-self-attention` | M2 |
| 90 | `qk-norm-attention` | QK-Norm Attention | easy | numeric | t/j/m | `causal-self-attention`, `rmsnorm` | M2 |
| 100 | `padded-batch-invariance` | Batching Without Changing the Answer | medium | numeric, property | t/j/m | `decoder-forward`, `attention-mask-builder` | M2 |
| 110 | `moe-router-forward` | Mixture-of-Experts Layer | medium | numeric | t/j/m | `swiglu-mlp` | M2 |
| 120 | `embedding-scale-softcap` | Embedding Scaling and Logit Soft-Capping | easy | numeric | t/j/m | - | M5 |
| 130 | `fused-projection-split` | Splitting Fused Projections | easy | numeric, property | t/j/m | `swiglu-mlp` | M5 |
| 140 | `rope-layout-conversion` | Interleaved versus Half-Split RoPE | medium | numeric, property | t/j/m | `rope` | M5 |
| 150 | `attention-sinks-softmax` | Attention with Learned Sinks | medium | numeric, property | t/j/m | `causal-self-attention` | M5 |
| 160 | `moe-sparse-dispatch` | Sparse Expert Dispatch | hard | numeric, cost | t/j/m | `moe-router-forward` | M5 |
| 170 | `online-softmax-attention` | Blockwise Attention with Online Softmax | hard | numeric, property | t/j/m | `causal-self-attention`, `stable-softmax` | M5 |
| 180 | `real-model-block` | One Block of a Real Model | medium | numeric | t/j/m | `grouped-query-attention`, `swiglu-mlp`, `rope` | M3 |
| 190 | `real-model-logits` | Matching a Real Model's Logits | hard | numeric | t/j/m | `real-model-block`, `decoder-forward` | M3 |

**`swiglu-mlp`** — Implement `swiglu(x, w_gate, w_up, w_down)` computing `(silu(x W_gate^T) * (x W_up^T)) W_down^T` with weights in `(out, in)` layout (Shazeer, 2020; the MLP of Llama, Qwen, Granite). Numeric. Mutants: activation on the up branch; GELU in place of SiLU (that is GeGLU, as in Gemma); weights applied untransposed (hidden behind square shapes, so tests use non-square).

**`grouped-query-attention`** — Implement `gqa(q, k, v, mask)` with `q` of shape `(B, H, T, d)` and `k`, `v` of shape `(B, H_kv, S, d)`, `H` a multiple of `H_kv`: query head `h` uses KV head `h // (H / H_kv)` (Ainslie et al., 2023). Property: equals standard attention on explicitly repeated K and V. Mutants: `tile` instead of `repeat_interleave` (wrong head grouping); averaging query heads; wrong for `H_kv == 1`.

**`pre-norm-block`** — Implement `block(x, params, mask)` as `x + attn(norm1(x))` then `x + mlp(norm2(x))`, composing kit `ops` for attention, RMSNorm, and SwiGLU. Goal: the residual wiring, which is where ported models silently go wrong. Numeric. Mutants: post-norm; second residual taken from the block input; one norm shared for both sublayers. [kit: `ops`, `decoder_params`]

**`decoder-forward`** — Implement `forward(params, cfg, tokens) -> logits` for the kit's modern decoder layout: embedding, `cfg.n_layers` pre-norm blocks with RoPE-rotated GQA and SwiGLU, final RMSNorm, tied output head. Numeric against `decoder_forward_ref`, plus the causality property (changing token `t` leaves logits before `t` unchanged). Mutants: missing final norm; RoPE applied to values; untied head read from a missing key defaulting to zeros; positions starting at 1. [kit: `decoder_params`]

**`last-position-head`** — Implement `next_token_logits(hidden, w_out)` for `hidden` of shape `(B, T, D)` returning `(B, V)` for the last position only. Speed: within 5x of the reference at `T = 512`, `V = 8192`, where projecting every position first is hundreds of times more work. Goal: prefill needs one row of logits. Mutants: full projection then slice (slow); first position; flattening batch and time.

**`sliding-window-attention`** — Implement `window_mask(T, window)` and attention under it: query `i` attends to keys `j` with `i - window < j <= i` (the local layers of Gemma 3; Mistral 7B). Property: with `window >= T` the result equals causal attention; tokens outside the window have zero influence. Mutants: window off by one; symmetric window that leaks the future; window measured from the sequence end.

**`qk-norm-attention`** — Implement attention where each head's queries and keys are RMS-normalized with learned gains over the head dimension before RoPE and the dot product (as in Qwen3 `q_norm`/`k_norm` and Gemma 3). Numeric. Mutants: normalizing over the full hidden size before the head split; normalizing after RoPE's scale; normalizing values too.

**`padded-batch-invariance`** — Implement `batched_forward(params, cfg, seqs, pad_id)` for sequences of different lengths: left-pad, build masks and per-row RoPE positions so each row's last-token logits equal the unpadded single-sequence result. Property judge over random length mixes. Mutants: positions counted from the padded start; pads attended; right padding with last-column readout. [kit: `decoder_params`]

**`moe-router-forward`** — Implement `moe(x, w_router, experts_w, k)`: per token, softmax over router logits, keep the top `k` experts (ties to the lowest index), renormalize their weights, and return the weighted sum of expert SwiGLU outputs (dense evaluation is acceptable here). Numeric. Mutants: no renormalization after top-k; softmax taken after top-k of logits with a different result under the specified order; one expert choice per sequence instead of per token.

**`embedding-scale-softcap`** — Implement `embed(tokens, table, scale)` and `finalize_logits(h, w_out, logit_scale, softcap)`: embedding multiplier (Gemma's `sqrt(d)`, Granite's 12), logit divisor, and optional soft-capping `cap * tanh(logits / cap)` (Gemma). Goal: the per-family constants that make a correct architecture produce wrong logits. Numeric. Mutants: scale applied to the tied head too; cap applied before the divisor; cap as a hard clip.

**`fused-projection-split`** — Implement `split_fused(w, parts)` and `fuse(ws)` for fused projections (QKV in one matrix; gate and up in Granite's `input_linear`), and `swiglu_fused(x, w_in, w_down)`. Property: the fused and split forms give equal outputs. Mutants: split along the input axis; halves swapped (gate and up exchanged, visible only through the activation).

**`rope-layout-conversion`** — Implement `rope_interleaved(x, pos, theta)` (pairs `(0,1), (2,3)`, the original Llama layout), `rope_half(x, pos, theta)` (pairs `(i, i + d/2)`, the Hugging Face layout), and `permute_qk(w, d_head, n_heads)` that converts projection weights from one layout to the other. Property: attention scores are identical after conversion. Mutants: permuting queries only; permuting across heads instead of within; inverse permutation.

**`attention-sinks-softmax`** — Implement `attention_with_sinks(q, k, v, mask, sinks)`: each head has a learned scalar sink logit added as an extra softmax column that receives probability mass but contributes no value (as in gpt-oss; the `sinks` argument of `mx.fast.scaled_dot_product_attention`). Property: rows of attention weights over real keys sum to less than 1, and to 1 as the sink goes to `-inf`. Mutants: renormalizing over real keys; sink scaled by `1/sqrt(d)`; one sink shared by all heads.

**`moe-sparse-dispatch`** — Implement `moe_sparse(x, w_router, experts, k)` where `experts` is a kit object: group tokens by expert and call each expert once on exactly its routed rows, then scatter back. Cost: `expert_rows == k * n_tokens` and `expert_calls <=` number of experts with any token. Numeric against the dense oracle. Mutants: dense evaluation; one call per token; scatter with the wrong row order when an expert receives non-contiguous tokens. [kit: `CountingExperts`]

**`online-softmax-attention`** — Implement `blockwise_attention(q, kv_blocks)` where `kv_blocks` is a one-pass iterator of `(k, v, mask)` blocks: maintain a running maximum, a running normalizer, and a rescaled running output so the result equals full attention without holding all scores (Milakov and Gimelshein, 2018; the core of FlashAttention, Dao et al., 2022). Property: the result is invariant to how keys are partitioned; numeric with score ranges that overflow a naive running sum. Mutants: no rescale when the maximum grows; fully masked block producing NaN; normalizer accumulated per block and averaged.

**`real-model-block`** — Implement `granite_block(x, weights, layer, positions)` for `granite-4.0-350m` from the raw safetensors tensors: RMSNorm (eps 1e-5), GQA with 16 heads over 4 KV heads, RoPE theta 1e7, attention scale `1/64`, fused SwiGLU input projection, residual multiplier 0.263. Numeric against fixture hidden states for layers 0 and 13 (`rtol` 1e-3). Mutants: scale `1/sqrt(64)`; residual multiplier omitted; gate and up halves swapped. [kit: `assets`]

**`real-model-logits`** — Implement `granite_logits(weights, cfg, tokens)`: the full 28-layer forward of `granite-4.0-350m` from a name-to-array dict, matching fixture logits produced by `transformers` on prompts of up to 16 tokens (`rtol` 1e-3, `atol` 1e-3) and the fixture argmax tokens exactly. Goal: porting a checkpoint with no reference code open. Mutants: embedding multiplier missing; logits divisor missing; final norm missing; BF16 weights used without conversion to float32. [kit: `assets`]

### Track 4: `sampling` Sampling and Decoding (order 40)

From logits to a token: the samplers production engines ship, and the order they run in.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `top-p-filtering` | Nucleus (Top-p) Filtering | medium | numeric, property | t/j/m | - | M1 |
| 20 | `repetition-penalty` | Vectorized Repetition Penalty | medium | numeric, speed | t/j/m | - | M1 |
| 30 | `temperature-scaling` | Temperature | easy | numeric, property | t/j/m | `stable-softmax` | M2 |
| 40 | `top-k-filtering` | Top-k Filtering | easy | numeric, property | t/j/m | - | M2 |
| 50 | `categorical-sample` | Inverse-CDF Sampling | easy | numeric, statistical | t/j/m | `stable-softmax` | M2 |
| 60 | `min-p-filtering` | Min-p Filtering | easy | numeric, property | t/j/m | - | M2 |
| 70 | `frequency-presence-penalty` | Frequency and Presence Penalties | easy | numeric | t/j/m | `repetition-penalty` | M2 |
| 80 | `sampler-chain` | A Sampler Chain in the Right Order | medium | numeric, statistical | t/j/m | `top-p-filtering`, `top-k-filtering`, `temperature-scaling` | M2 |
| 90 | `stop-sequences` | Stop Sequences Across Token Boundaries | medium | numeric, property | python | `streaming-detokenizer` | M2 |
| 100 | `beam-search` | Beam Search with a KV Cache | hard | numeric, cost | t/j/m | `cached-greedy-decode` | M2 |
| 110 | `top-n-sigma` | Top-n-sigma Filtering | easy | numeric | t/j/m | - | M5 |
| 120 | `gumbel-max-sample` | Gumbel-Max Sampling | medium | numeric, statistical | t/j/m | `categorical-sample` | M5 |
| 130 | `logit-bias-bad-words` | Logit Bias and Banned Sequences | medium | numeric | t/j/m | - | M5 |
| 140 | `xtc-sampler` | XTC: Exclude Top Choices | medium | numeric, statistical | t/j/m | `min-p-filtering` | M5 |
| 150 | `mirostat-v2` | Mirostat v2 | medium | numeric, property | t/j/m | `categorical-sample` | M5 |
| 160 | `dry-penalty` | DRY Repetition Penalty | hard | numeric, speed | python | `repetition-penalty` | M5 |

**`temperature-scaling`** — Implement `apply_temperature(logits, temperature)` with per-row temperatures of shape `(B,)`; a temperature of exactly 0 is specified as greedy (the argmax keeps logit 0, all others `-inf`, ties to the lowest index). Property: ordering preserved; entropy monotone in temperature. Mutants: division by zero giving NaN; multiplying instead of dividing; `-inf` entries turned into NaN.

**`top-k-filtering`** — Implement `top_k_filter(logits, k)` keeping exactly `k` entries per row (ties at the threshold broken toward the lowest index), `k >= V` a no-op, masked entries staying masked. Property: exactly `min(k, finite)` survivors. Mutants: `>= threshold` keeping extra tied entries; `k = 0` crash (specified as no-op, the llama.cpp convention); unstable sort.

**`categorical-sample`** — Implement `sample(probs, u)` for a batch `(B, V)` and uniforms `(B,)`: the smallest index whose cumulative probability exceeds `u`, never returning a zero-probability token even when float32 cumulative sums fall short of 1. Statistical judge on the distribution, numeric on worked cases. Mutants: `>=` versus `>` returning zero-probability tokens at `u = 0`; cumulative sum ending at 0.9999 leading to an out-of-range index; one `u` shared across the batch.

**`min-p-filtering`** — Implement `min_p_filter(logits, min_p)`: keep tokens whose probability is at least `min_p` times the top probability (Nguyen et al., 2024; a default sampler in llama.cpp and mlx-lm). Property: the top token always survives. Mutants: threshold as an absolute probability; threshold applied to logits; comparison done after renormalizing the survivors.

**`frequency-presence-penalty`** — Implement `apply_penalties(logits, token_counts, frequency_penalty, presence_penalty)` with the OpenAI definition: subtract `count * frequency + (count > 0) * presence`. Numeric. Mutants: multiplicative form (that is repetition penalty); presence scaled by count; counts taken from the prompt when the spec says generated tokens only.

**`sampler-chain`** — Implement `sample_chain(logits, cfg, history, u)` applying, in llama.cpp's default order, penalties, top-k, top-p, min-p, then temperature, then inverse-CDF sampling; each stage from kit `ops`. Goal: order matters, since temperature before top-p changes the nucleus. Numeric on filtered logits, statistical on samples from a config where the two orders give different supports. Mutants: temperature first; top-p computed on unnormalized survivors of top-k; penalties after filtering. [kit: `ops`]

**`stop-sequences`** — Implement `stop_step(state, piece, stops) -> (state, emit, stopped)`: stream text pieces through, holding back the longest suffix that is a proper prefix of any stop string, cutting output exactly before the earliest stop match, and `stop_flush` releasing held text at end of stream. Property: emitted text concatenates to the full text truncated at the first stop, for every chunking. Mutants: checks each piece alone; emits the partial stop then cannot retract; holds back a fixed `max(len(stop))` characters forever.

**`beam-search`** — Implement `beam_search(lm, prompt, n_beams, max_new_tokens, eos_id, length_penalty)`: expand all live beams in one batched model call per step, reorder caches with `cache.select`, move finished hypotheses to a done set scored `logprob / len**length_penalty`, and stop when no live beam can beat the worst finished one. Cost: `forward_calls <= max_new_tokens`, `tokens_processed <= P + n_beams * (max_new_tokens - 1)`. Mutants: cache not reordered after beam reshuffle; finished beams kept expanding; scores normalized before pruning. [kit: `KVCache.select`]

**`top-n-sigma`** — Implement `top_n_sigma_filter(logits, n)`: keep tokens with logit at least `max - n * std`, where the standard deviation is over finite logits (Tang et al., 2024; in llama.cpp's default chain). Numeric. Mutants: std over probabilities; masked `-inf` included in the std (NaN); mean in place of max.

**`gumbel-max-sample`** — Implement `gumbel_sample(logits, uniforms)` with one uniform per vocabulary entry: `argmax(logits - log(-log(u)))`, stable at `u` near 0 and 1, masked entries never chosen. Statistical judge: samples follow softmax of the logits. Mutants: one `log`; noise added to probabilities; `-inf + inf` NaN on masked entries.

**`logit-bias-bad-words`** — Implement `apply_bias(logits, bias, allowed_token_ids, bad_words, history)`: additive bias, an allow-list, and banning of multi-token sequences by masking a sequence's last token only when the history ends with its preceding tokens (the `bad_words` semantics of vLLM and transformers). Numeric. Mutants: bans every token of a bad sequence; prefix match ignoring order; bias applied to disallowed tokens making them finite again.

**`xtc-sampler`** — Implement `xtc_filter(logits, threshold, probability, u)`: when `u < probability` and at least two tokens have probability `>= threshold`, remove all of them except the least likely one (XTC by p-e-w, as merged in llama.cpp and mlx-lm). Statistical check on how often the top token is removed. Mutants: removes all above-threshold tokens; removes the top one only; fires when a single token is above threshold.

**`mirostat-v2`** — Implement `mirostat_step(logits, mu, tau, eta, u) -> (token, mu)`: truncate tokens whose surprise exceeds `mu`, sample from the rest, and update `mu -= eta * (surprise - tau)` (Basu et al., 2020). Property: over a long seeded run on stationary logits, mean surprise converges to `tau` within a stated band. Mutants: natural log where the spec uses bits; update with the wrong sign; surprise measured after renormalization.

**`dry-penalty`** — Implement `dry_penalties(history, multiplier, base, allowed_length, breakers) -> dict[token, penalty]`: for every token that would extend a repeat of an earlier sequence, penalty `multiplier * base**(match_len - allowed_length)`, with sequence breakers resetting matches (DRY by p-e-w, as in llama.cpp). Speed: linear time via the Z-algorithm on the reversed history, within 5x of the reference on 20,000 tokens. Mutants: quadratic scan; longest match not taken per token; breakers ignored.

### Track 5: `kv-cache` KV Cache and Attention Memory (order 50)

Never compute a token twice, and never keep more memory than the answer needs.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `cached-greedy-decode` | Greedy Decoding with a KV Cache | medium | cost, numeric | t/j/m | - | M1 |
| 20 | `kv-cache-append` | Attention with an Appended Cache | easy | numeric, property | t/j/m | `causal-self-attention` | M2 |
| 30 | `chunked-prefill-cache` | Prefill in Chunks | easy | numeric, cost | t/j/m | `cached-greedy-decode` | M2 |
| 40 | `cached-rope-offsets` | RoPE Positions Under a Cache | easy | numeric, property | t/j/m | `rope`, `kv-cache-append` | M2 |
| 50 | `preallocated-kv-cache` | A Preallocated KV Cache | medium | numeric, property | t/j/m | `kv-cache-append` | M2 |
| 60 | `batched-cached-decode` | Batched Decoding with Early Finish | medium | numeric, cost | t/j/m | `cached-greedy-decode` | M2 |
| 70 | `prefix-cache-reuse` | Reusing a Cached Prefix | medium | numeric, cost | t/j/m | `cached-greedy-decode` | M2 |
| 80 | `sliding-window-cache` | A Rotating Cache That Keeps Its Sinks | medium | numeric, property | t/j/m | `preallocated-kv-cache` | M2 |
| 90 | `paged-kv-allocator` | A Paged KV Block Allocator | easy | numeric, property | python | - | M2 |
| 100 | `paged-kv-copy-on-write` | Forking Sequences with Copy-on-Write | hard | numeric, property | python | `paged-kv-allocator` | M5 |
| 110 | `block-hash-prefix-cache` | Automatic Prefix Caching by Block Hash | medium | numeric, property | python | `paged-kv-allocator`, `prefix-cache-reuse` | M5 |
| 120 | `radix-prefix-cache` | A Radix-Tree Prefix Cache | hard | numeric, property | python | `prefix-cache-reuse` | M5 |
| 130 | `kv-cache-quantize` | An 8-bit KV Cache | medium | numeric, property | t/j/m | `affine-int8`, `kv-cache-append` | M5 |
| 140 | `kv-eviction-heavy-hitters` | Heavy-Hitter KV Eviction | medium | numeric, property | t/j/m | `kv-cache-append` | M5 |
| 150 | `snapkv-compress` | Compressing a Prompt Cache by Attention Votes | medium | numeric | t/j/m | `kv-eviction-heavy-hitters` | M5 |
| 160 | `real-model-cached-generate` | Cached Generation on a Real Model | medium | numeric, speed | t/j/m | `kv-cache-append`, `cached-rope-offsets`, `real-model-logits` | M3 |

**`kv-cache-append`** — Implement `attend_step(q_new, k_new, v_new, cache) -> (out, cache)` where `cache` is a `(k, v)` tuple or `None`: append the new keys and values and attend from the new queries over everything, with causality among the new tokens. Property: feeding a sequence in any chunking equals one full causal pass. Mutants: no causal mask within a multi-token chunk; new tokens attend only to the cache; append on the head axis.

**`chunked-prefill-cache`** — Implement `prefill(lm, prompt, chunk_size) -> (last_logits, cache)` feeding the prompt in consecutive chunks of at most `chunk_size` tokens (mlx-lm `prefill_step_size`; the single-sequence core of chunked prefill). Cost: `forward_calls == ceil(P / chunk_size)`, `tokens_processed == P`. Mutants: re-feeding the prefix with each chunk; a trailing empty call; returning the first chunk's logits.

**`cached-rope-offsets`** — Implement `rope_positions(cache_lengths, n_new)` returning per-row absolute positions for the new tokens, and `rotate_new(q, k, positions, theta)` applying RoPE to new queries and keys only (cached keys are stored rotated). Property: chunked equals full. Mutants: positions restart at 0 each call; cached keys rotated again; one offset for the whole batch. [kit: `ops`]

**`preallocated-kv-cache`** — Implement `cache_init(capacity, ...)`, `cache_write(cache, k_new, v_new) -> cache`, and `cache_attend(q, cache)` over a fixed-capacity buffer with a length field: buffer shapes never change, and the unused tail never affects the output. Tests fill the tail with NaN and large values; writing past capacity raises `ValueError`. Mutants: attends over the whole buffer; concatenates (shape grows); off-by-one length after a multi-token write.

**`batched-cached-decode`** — Implement `batched_greedy(lm, prompts, max_new_tokens, eos_id)` for equal-length prompts `(B, P)`: one batched call per step, per-row EOS tracking, finished rows fed `eos_id` and ignored, loop ends as soon as every row has finished. Cost: `forward_calls <= longest row`, and no call after all rows finish. Mutants: stops when any row finishes; never stops early; tokens after EOS returned.

**`prefix-cache-reuse`** — Implement `generate_with_store(lm, store, prompt, max_new_tokens) -> (tokens, store)` where `store` maps token tuples to caches: find the entry with the longest common token prefix, truncate its cache to that prefix (keeping at least one prompt token to feed, so logits exist), process only the rest, and store the new prompt's cache. Cost: `tokens_processed <= P - reused + g - 1`. Mutants: reuse only on exact match; identical prompt feeds zero tokens and crashes; cache not truncated when prompts diverge mid-entry. [kit: `KVCache.truncate`]

**`sliding-window-cache`** — Implement `rotating_write(cache, k_new, v_new, max_size, keep)`: once full, overwrite the oldest entries after the first `keep` tokens, and return keys in temporal order for attention (mlx-lm `RotatingKVCache`; attention sinks of StreamingLLM, Xiao et al., 2023). Property: contents always equal "first `keep` plus most recent `max_size - keep`". Mutants: evicts the sink tokens; order scrambled after wraparound; multi-token write crossing the wrap point.

**`paged-kv-allocator`** — Implement `BlockAllocator(n_blocks, block_size)` with `append(seq_id, n_tokens)`, `free(seq_id)`, `block_table(seq_id)`, `free_blocks()`: sequences own lists of fixed-size blocks, allocated only when the last block is full, lowest free id first (PagedAttention, Kwon et al., 2023). Raises `OutOfBlocks` without partial allocation. Property over random op sequences: no block owned twice, no leak. Mutants: allocates a block per token; partial allocation on failure; free list not restored.

**`paged-kv-copy-on-write`** — Extend the allocator with `fork(parent, child)` sharing all blocks by reference count, and copy-on-write when either sequence appends into a shared, partially filled last block; return the list of `(src, dst)` block copies the engine must perform. Property: reference counts equal the number of owners; forked siblings never see each other's appends. Mutants: copies every block on fork; no copy on a shared last block; double free when the parent finishes first.

**`block-hash-prefix-cache`** — Implement `PrefixBlockCache(block_size, capacity)`: `lookup(tokens)` returns the number of leading full blocks already cached, where a block's key is the hash of `(parent_key, block_tokens)`; `insert`, reference counting for in-use blocks, and LRU eviction among unreferenced blocks (vLLM automatic prefix caching). Mutants: hash of the block alone (false hits on a shared middle block); partial last block cached; evicts a referenced block.

**`radix-prefix-cache`** — Implement `RadixCache` over token sequences: `match_prefix(tokens) -> (length, node)`, `insert(tokens)` splitting an edge at the divergence point, lock counts on the path of a running request, and `evict(n_tokens)` removing least-recently-used unlocked leaves (SGLang RadixAttention, Zheng et al., 2024). Property against a brute-force set of stored sequences. Mutants: no edge split (match stops at node boundaries); evicts an inner node with children; access time not propagated to ancestors.

**`kv-cache-quantize`** — Implement `quantized_append(cache, k_new, v_new, bits, group_size, keep_recent)`: tokens older than `keep_recent` are stored as group-wise affine codes with per-group scale and offset, recent tokens stay float (mlx-lm `QuantizedKVCache`, `quantized_kv_start`). Property: reconstruction error at most half a step per element; attention output within stated tolerance of the float cache. Mutants: re-quantizing already quantized tokens on every append (error accumulates); one scale for the whole cache; `keep_recent` off by one.

**`kv-eviction-heavy-hitters`** — Implement `h2o_step(cache, scores, attn_weights, budget, recent)`: add this step's attention weights to per-token running scores; when over budget, keep the `recent` newest tokens and the highest-scoring older ones, preserving temporal order (H2O, Zhang et al., 2023). Numeric against the oracle's kept indices. Mutants: evicts by lowest latest-step attention; scores not carried through eviction; order not preserved.

**`snapkv-compress`** — Implement `snapkv_select(attn, window, budget, kernel)`: average the attention of the last `window` query tokens over prefix positions per head, max-pool with `kernel` to keep neighbors, and keep the top `budget - window` prefix positions plus the window (SnapKV, Li et al., 2024). Numeric on selected indices per head. Mutants: no pooling; votes averaged across heads when the spec is per head; window tokens counted inside the budget twice.

**`real-model-cached-generate`** — Implement `generate(weights, cfg, prompt_ids, max_new_tokens)` for `granite-4.0-350m` with your own KV cache: prefill once, then one token per step through all 28 layers, GQA keys and values cached at KV-head width. Judged on exact fixture token ids, and on speed: within 3x of the cached reference for 16 prompt and 24 generated tokens, where re-running the prefix each step is more than 10x slower. Mutants: no cache (slow); RoPE positions restart; keys cached before rotation and rotated again each step. [kit: `assets`]

### Track 6: `constrained` Constrained Generation (order 60)

Guaranteeing the shape of the output by masking the tokens that would break it.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `token-mask-apply` | Applying a Token Mask | easy | numeric, property | t/j/m | `stable-softmax` | M2 |
| 20 | `choice-trie-constraint` | Constraining Output to a Set of Choices | easy | numeric | python | - | M2 |
| 30 | `dfa-token-mask` | Token Masks from a Character DFA | medium | numeric, property | python | - | M2 |
| 40 | `dfa-token-index` | Precomputing the Token Index | medium | numeric, cost | python | `dfa-token-mask` | M2 |
| 50 | `json-pushdown-validator` | An Incremental JSON Validator | medium | numeric, property | python | - | M2 |
| 60 | `constrained-greedy-decode` | Constrained Decoding Loop | medium | numeric, cost | t/j/m | `cached-greedy-decode`, `token-mask-apply` | M2 |
| 70 | `regex-to-dfa` | Compiling a Regex to a DFA | hard | numeric, property | python | `dfa-token-mask` | M5 |
| 80 | `json-schema-regex` | JSON Schema to Regex | hard | numeric, property | python | `regex-to-dfa` | M5 |
| 90 | `jump-forward-decode` | Jump-Forward Decoding | medium | numeric, cost | t/j/m | `constrained-greedy-decode` | M5 |
| 100 | `token-healing` | Token Healing | medium | numeric | python | `bpe-encode` | M5 |
| 110 | `tool-call-stream-parser` | Parsing Tool Calls from a Stream | medium | numeric, property | python | `json-pushdown-validator` | M5 |
| 120 | `grammar-token-mask` | Token Masks for a Context-Free Grammar | hard | numeric, property | python | `dfa-token-mask`, `json-pushdown-validator` | M5 |
| 130 | `structural-tag-dispatch` | Triggered Constraints with Structural Tags | medium | numeric, property | python | `constrained-greedy-decode` | M5 |

**`token-mask-apply`** — Implement `mask_logits(logits, allowed)` for a boolean `(B, V)` mask (disallowed to `-inf`) and `masked_probs(logits, allowed)`; a row with nothing allowed raises `ValueError` rather than returning NaN. Property: disallowed probability is exactly 0 and the rest renormalize. Mutants: large negative constant; multiply probabilities by the mask without renormalizing; all-masked row returns NaN.

**`choice-trie-constraint`** — Implement `build_trie(choices_token_ids)` and `allowed_next(trie, generated)` returning the sorted token ids that keep the output a prefix of some choice, with `EOS` allowed exactly when the output equals a choice (vLLM `choice`). Numeric. Mutants: EOS allowed at any prefix; a choice that is a prefix of another cannot be completed; unsorted or duplicated ids.

**`dfa-token-mask`** — Implement `allowed_tokens(dfa, state, vocab)` and `advance(dfa, state, token)` by walking each token's bytes through a `CharDFA`: a token is allowed if the walk never dies (it need not end in an accepting state). Property: any allowed sequence decodes to a string the regex accepts as a prefix. Mutants: requires acceptance after each token; checks only the token's first byte; empty-bytes special tokens always allowed. [kit: `CharDFA`, `ToyVocab`]

**`dfa-token-index`** — Implement `build_index(dfa, vocab) -> index` and `allowed_from_index(index, state)`, mapping each DFA state to `{token: next_state}` ahead of time (Willard and Louf, 2023; Outlines). Cost: after `build_index`, mask queries and advances make zero `dfa.counters.steps`; building walks each (state, token) pair at most once and shares work across tokens with a common prefix through a vocabulary trie (a stated step bound that the naive per-token walk exceeds). Mutants: lazy walk at query time; index keyed by token only; dead states stored. [kit: `CharDFA`]

**`json-pushdown-validator`** — Implement `json_step(state, ch) -> state` (raising `Invalid`), `json_complete(state)`, and `json_start()`: a character-level pushdown recognizer for RFC 8259 JSON with a container stack, usable as a prefix validator. Property: accepts every prefix of valid documents and rejects fixture near-misses (trailing commas, leading zeros, bare control characters, lone surrogates in `\u` escapes are allowed as the RFC says). Mutants: no stack (mismatched brackets); numbers like `01`; complete at depth 0 before any value.

**`constrained-greedy-decode`** — Implement `constrained_greedy(lm, prompt, constraint, max_new_tokens, eos_id)`: each step masks logits to `constraint.allowed(state)`, permits EOS only when `is_accepting`, and stops on EOS or when nothing is allowed. Cost limits as in `cached-greedy-decode`. Mutants: constraint state not advanced; EOS allowed mid-structure; mask applied after argmax (retry loop that costs extra forward calls). [kit: `TokenConstraint`, `ToyVocab`]

**`regex-to-dfa`** — Implement `compile_regex(pattern) -> (transitions, start, accepting)` for literals, escapes, character classes with ranges and negation, `.`, `|`, grouping, and `* + ? {m,n}`, by Thompson construction and subset construction over bytes. Property: agrees with Python `re.fullmatch` on seeded random strings from the pattern's alphabet. Mutants: `{m,n}` off by one; alternation binding tighter than concatenation; negated class including bytes outside ASCII incorrectly.

**`json-schema-regex`** — Implement `schema_to_regex(schema)` for a stated subset (object with required and optional properties in declaration order, string with `enum`, `minLength`/`maxLength`, integer, number, boolean, null, array with `items`, `minItems`/`maxItems`, nesting) producing a pattern for compact JSON (the Outlines approach). Judged by `re.fullmatch` on positive and negative fixture documents and seeded generated instances. Mutants: optional property leaves a dangling comma; string escapes unhandled; `maxItems` off by one.

**`jump-forward-decode`** — Implement `jump_forward_decode(lm, prompt, constraint, max_new_tokens, eos_id)`: whenever the constraint allows exactly one token, append it without consulting logits, and feed each run of forced tokens to the model in a single call with the next free token's predecessor (SGLang jump-forward; llguidance fast-forward tokens). Cost: `forward_calls` at most the number of free choices plus one. Mutants: one call per forced token; forced tokens never fed to the cache (wrong later logits); stops jumping at accepting states incorrectly. [kit: `TokenConstraint`]

**`token-healing`** — Implement `heal(prompt_ids, vocab) -> (trimmed_ids, required_prefix)` and `healing_allowed(required_prefix, generated_bytes, vocab)`: remove the prompt's last token and constrain generation so its bytes are regenerated as a prefix of the first generated tokens (Guidance token healing). Numeric. Mutants: only tokens equal to the removed token allowed; constraint not lifted once the prefix is covered; special tokens healed.

**`tool-call-stream-parser`** — Implement `parse_step(state, piece) -> (state, events)` consuming text pieces that may contain `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks and emitting OpenAI-style events: `content` deltas, `tool_call_start(index, name)`, `arguments` deltas, `tool_call_end`. Property: events are identical for every chunking of the same text. Mutants: tag split across pieces leaks as content; argument deltas re-emit from the start; a second call reuses index 0.

**`grammar-token-mask`** — Implement `GrammarMatcher(grammar, vocab)` with `allowed(state)` and `advance(state, token)` for a GBNF subset (rules, literals, character classes, alternation, repetition, recursion) using a pushdown automaton with a set of stacks, walking a vocabulary trie so shared token prefixes are parsed once (XGrammar, Dong et al., 2024; llguidance). Property against the kit oracle `g.accepts_prefix` on every allowed and a sample of disallowed tokens. Mutants: single stack (wrong on ambiguity); left recursion loops; token spanning two rules rejected. [kit: `Grammar`, `ToyVocab`]

**`structural-tag-dispatch`** — Implement `TagDispatch(triggers)` as a `TokenConstraint`: free text until a trigger string (for example `<tool_call>`) has been generated, then the tag's inner constraint until its end string, then free text again; several tags may share a trigger prefix (structural tags in vLLM; TagDispatch in XGrammar-2, Li et al., 2026). Property across token boundaries that split the trigger. Mutants: trigger recognized only when it is a single token; inner constraint not reset for a second call; free text allowed to contain a half-open trigger at EOS. [kit: `TokenConstraint`]

### Track 7: `quantization` Quantization and Compression (order 70)

Fewer bits per weight, with the error measured rather than hoped for.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `affine-int8` | Affine int8 Quantization | medium | numeric, property | t/j/m | - | M1 |
| 20 | `symmetric-per-channel` | Symmetric Per-Channel Weights | easy | numeric, property | t/j/m | `affine-int8` | M2 |
| 30 | `quant-error-metrics` | Measuring Quantization Error | easy | numeric | t/j/m | - | M2 |
| 40 | `bit-packing` | Packing 4-bit Codes | easy | numeric, property | t/j/m | - | M2 |
| 50 | `groupwise-4bit` | Group-Wise 4-bit Quantization | medium | numeric, property | t/j/m | `affine-int8` | M2 |
| 60 | `quantized-matmul` | Multiplying Without Dequantizing | medium | numeric, property | t/j/m | `groupwise-4bit` | M2 |
| 70 | `clipping-calibration` | Calibrating a Clipping Range | medium | numeric, property | t/j/m | `affine-int8`, `quant-error-metrics` | M2 |
| 80 | `lora-merge` | Merging a LoRA Adapter | easy | numeric, property | t/j/m | - | M2 |
| 90 | `ternary-bitnet` | Ternary Weights | easy | numeric, property | t/j/m | - | M5 |
| 100 | `int8-linear-w8a8` | An Integer-Only Linear Layer | medium | numeric | t/j/m | `symmetric-per-channel` | M5 |
| 110 | `nf4-codebook` | NormalFloat 4-bit | medium | numeric | t/j/m | `groupwise-4bit` | M5 |
| 120 | `mxfp4-block-encode` | Microscaling FP4 Blocks | medium | numeric, property | t/j/m | `groupwise-4bit` | M5 |
| 130 | `bit-allocation` | Mixed-Precision Bit Allocation | medium | numeric, property | python | `quant-error-metrics` | M5 |
| 140 | `awq-scale-search` | Activation-Aware Scaling | medium | numeric, property | t/j/m | `groupwise-4bit`, `quant-error-metrics` | M5 |
| 150 | `k-quant-superblock` | GGUF K-Quant Superblocks | hard | numeric, property | t/j/m | `groupwise-4bit`, `bit-packing` | M5 |
| 160 | `gptq-column-update` | GPTQ Error Propagation | hard | numeric, property | t/j/m | `symmetric-per-channel` | M5 |
| 170 | `quantize-real-weights` | Quantizing Real Weights | medium | numeric | t/j/m | `groupwise-4bit`, `quant-error-metrics` | M3 |
| 180 | `mlx-quantized-checkpoint-load` | Reading an 8-bit Checkpoint | medium | numeric | python | `bit-packing`, `safetensors-parse` | M3 |

**`symmetric-per-channel`** — Implement `quantize_weight(w, bits)` returning int8 codes and one scale per output row, `scale = max|row| / (2**(bits-1) - 1)`, zero point fixed at 0, and `dequantize`. Property: error at most half a step per element; an all-zero row does not divide by zero. Mutants: one scale for the tensor; range `[-128, 127]` asymmetric (clips `+max`); scale per input column.

**`quant-error-metrics`** — Implement `mse(a, b)`, `sqnr_db(ref, approx)`, `max_abs_error`, and `output_error(w, w_hat, x)` (relative error of `x w^T`). Numeric. Mutants: SQNR with `20 log10` on power ratios; error normalized by the approximation; mean over the wrong axis.

**`bit-packing`** — Implement `pack4(codes)` (uint8 values below 16, length padded to a multiple of 8) into uint32 words, little-endian nibble order, lowest index in the lowest bits, and `unpack4(words, n)` (the layout MLX uses for 4-bit weights; verify against `mx.quantize` output when authoring). Round-trip property. Mutants: high nibble first; sign extension on shift; padding nibbles returned.

**`groupwise-4bit`** — Implement `quantize_groups(w, bits, group_size)` returning codes plus per-group `scale` and `bias` over consecutive groups of `group_size` along the input axis, `w ~ code * scale + bias`, with a constant group handled (the affine mode of `mx.quantize`, default `group_size=64`). Property: error at most half a step; numeric against the oracle. Mutants: groups along the output axis; min and max over the whole row; scale zero on a constant group (NaN).

**`quantized-matmul`** — Implement `qmatmul(x, codes, scale, bias, group_size)` computing `x @ dequant(W)^T` as `sum over groups of scale * (x_g @ codes_g^T) + bias * sum(x_g)` without building the float weight matrix (what `mx.quantized_matmul` does). Numeric and property against dequantize-then-multiply over random group sizes and non-square shapes; the statement forbids building the float weight matrix, and the M3 and M4 kernel problems enforce it by memory. Mutants: bias term dropped; bias multiplied by group size instead of `sum(x_g)`; groups misaligned with `x`.

**`clipping-calibration`** — Implement `best_clip(samples, bits, candidates)`: for each candidate clipping fraction of the absolute maximum, quantize the calibration samples symmetrically and return the candidate with the lowest mean squared error, ties to the larger range. Property: never worse than no clipping on the calibration set. Mutants: minimizes clipping error only; percentile taken per sample; ties to the smaller range.

**`lora-merge`** — Implement `merge(w, a, b, alpha)` returning `w + (alpha / r) * b @ a` with `a: (r, in)`, `b: (out, r)`, `unmerge`, and `lora_forward(x, w, a, b, alpha)` (Hu et al., 2021). Property: merged forward equals unmerged forward; merge then unmerge is the identity within tolerance. Mutants: `a @ b` order; scale `alpha` without `/ r`; `alpha / r` applied twice.

**`ternary-bitnet`** — Implement `ternary_quantize(w)` returning values in `{-1, 0, 1}` and one scale `mean(|w|)`, with `round(clip(w / scale, -1, 1))` (BitNet b1.58, Ma et al., 2024), and `ternary_matmul(x, t, scale)` using only additions and subtractions of selected columns. Mutants: threshold at `max|w|`; scale from non-zero entries only; zero mapped to +1.

**`int8-linear-w8a8`** — Implement `int8_linear(x, w_q, w_scale, bias)`: quantize activations dynamically per token to int8 (symmetric, one scale per row of `x`), accumulate `x_q @ w_q^T` in int32, then rescale by `x_scale * w_scale` (the W8A8 dynamic per-token scheme of llm-compressor). Numeric against the float layer within the stated bound, and an exact check of the int32 accumulator. Mutants: int8 accumulator overflow; one activation scale per batch; bias added before rescaling.

**`nf4-codebook`** — Implement `nf4_levels()` (the 16 NormalFloat levels from the quantiles of a standard normal, as constructed in QLoRA, Dettmers et al., 2023, with an exact zero), `nf4_quantize(w, block_size)` (per-block absmax normalization, nearest level), and `nf4_dequantize`. Numeric against the published bitsandbytes level table (stated in the problem). Mutants: uniform levels; nearest by index; normalization by standard deviation.

**`mxfp4-block-encode`** — Implement `mxfp4_encode(x)` and `mxfp4_decode(codes, exps)` for blocks of 32 values sharing one 8-bit power-of-two scale, each element a 4-bit E2M1 float (OCP Microscaling Formats v1.0, 2023; the `mxfp4` mode of `mx.quantize`). Property: decode of encode is the nearest representable value under the chosen shared exponent; the exponent is the spec's `floor(log2(max|x|)) - emax`. Mutants: per-element exponent; round toward zero; subnormal element values ignored.

**`bit-allocation`** — Implement `allocate_bits(layers, options, target_bpw)`: each layer has a parameter count and an error for each bit option; choose one option per layer minimizing total error subject to average bits per weight at most the target, by the specified exact dynamic program over a discretized budget (the problem behind `mlx_lm.dynamic_quant` and mixed GGUF recipes). Property: no feasible assignment has lower error (brute force on small cases). Mutants: greedy by error per bit; average unweighted by layer size; budget off by rounding.

**`awq-scale-search`** — Implement `awq_scales(w, x_calib, bits, group_size, grid)`: per input channel scale `s = mean|x|**alpha` normalized as specified, search `alpha` over the grid to minimize `||x w^T - (x / s) Q(w * s)^T||`, return `s` and the best `alpha` (AWQ, Lin et al., 2023). Property: output error not worse than `alpha = 0`. Mutants: scale applied to weights only; error measured on weights instead of outputs; activation mean signed.

**`k-quant-superblock`** — Implement `q4k_encode(w)` and `q4k_decode(...)` for a simplified but exactly specified Q4_K layout: superblocks of 8 blocks of 32 weights, 4-bit codes, per-block scale and minimum each quantized to 6 bits against a per-superblock float16 scale and minimum (GGUF K-quants, llama.cpp). Property: decode error bounded by the two-level step sizes; codes use exactly 4.5 bits per weight. Mutants: block minimum not clamped non-positive as the format requires; 6-bit fields rounded down; superblock scale from the first block.

**`gptq-column-update`** — Implement `gptq_quantize(w, h_inv_chol, quantize_col)`: process columns left to right, quantize column `j`, and propagate the scaled error to the remaining columns using row `j` of the upper Cholesky factor of the inverse Hessian (GPTQ, Frantar et al., 2022, Algorithm 1 without lazy batching). Property: output error on the calibration Hessian is lower than round-to-nearest for seeded cases; numeric against the oracle. Mutants: error divided by the wrong diagonal entry; update applied to already quantized columns; plain rounding.

**`quantize-real-weights`** — Implement `quantize_report(weights, bits, group_size)` over the linear layers of `granite-4.0-350m`: group-wise quantize each, and return per-layer SQNR, total bits per weight including scales and biases, and the three worst layers. Judged against fixture values at 8 and 4 bits (`rtol` 1e-3), plus a kit-computed logit KL divergence on a fixture prompt that must fall under stated limits. Mutants: bpw ignoring scale and bias overhead; embeddings skipped silently; SQNR averaged in dB across layers of different size without weighting as specified. [kit: `assets`]

**`mlx-quantized-checkpoint-load`** — Implement `load_quantized(buf, name) -> float32 array` for `lmstudio-community/granite-4.2-3b-MLX-8bit`: read `<name>.weight` (packed uint32), `<name>.scales`, `<name>.biases` from the safetensors file through a memory map, unpack 8-bit codes, and dequantize with group size 64. Judged on three small named tensors against fixture checksums and on the mean error versus fixture rows of the BF16 original. Mutants: byte order within the uint32 word; scales broadcast along the wrong axis; reading the whole 3.89 GB file (cost: bytes read through `CountingFile`). [kit: `assets`, `CountingFile`]

### Track 8: `speculative` Speculative and Accelerated Decoding (order 80)

Several tokens per pass of the big model, without changing what it would have said.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `speculative-sampling-step` | Speculative Sampling: One Verification Step | hard | statistical, numeric | t/j/m | - | M1 |
| 20 | `greedy-speculative-verify` | Verifying a Greedy Draft | easy | numeric | t/j/m | - | M2 |
| 30 | `prompt-lookup-draft` | Drafting by Prompt Lookup | easy | numeric | python | - | M2 |
| 40 | `speculative-speedup-model` | Expected Speedup of Speculation | easy | numeric | python | - | M2 |
| 50 | `speculative-sampling-block` | Verifying a Block of Sampled Drafts | medium | statistical, numeric | t/j/m | `speculative-sampling-step` | M2 |
| 60 | `speculative-decode-loop` | A Speculative Decoding Loop | hard | numeric, cost | t/j/m | `greedy-speculative-verify`, `cached-greedy-decode` | M2 |
| 70 | `dynamic-draft-length` | Adapting the Draft Length | medium | numeric, cost | python | `speculative-decode-loop` | M5 |
| 80 | `typical-acceptance` | Lossy Verification by Typical Acceptance | easy | numeric | t/j/m | `greedy-speculative-verify` | M5 |
| 90 | `batched-speculative-accept` | Ragged Acceptance in a Batch | medium | numeric, property | t/j/m | `greedy-speculative-verify` | M5 |
| 100 | `mtp-head-draft` | Drafting with Multi-Token Prediction Heads | medium | numeric | t/j/m | `greedy-speculative-verify` | M5 |
| 110 | `tree-attention-mask` | Verifying a Token Tree in One Pass | medium | numeric, property | t/j/m | `attention-mask-builder`, `greedy-speculative-verify` | M5 |
| 120 | `multi-draft-rejection` | Exact Sampling from Several Drafts | hard | statistical, numeric | t/j/m | `speculative-sampling-step` | M5 |
| 130 | `early-exit-self-speculation` | Self-Speculation by Early Exit | hard | numeric, cost | t/j/m | `speculative-decode-loop` | M5 |
| 140 | `suffix-automaton-draft` | Drafting from a Suffix Automaton | hard | numeric, speed | python | `prompt-lookup-draft` | M5 |

**`greedy-speculative-verify`** — Implement `verify_greedy(draft_tokens, target_logits) -> (n_accepted, next_token)` where `target_logits[i]` scores the position of `draft_tokens[i]` and there is one extra row: accept the longest prefix on which the target argmax equals the draft, and return the target's token at the first mismatch (or the bonus token after a full match). Numeric. Mutants: accepts matches after the first mismatch; no bonus token; off-by-one alignment of logits and drafts.

**`prompt-lookup-draft`** — Implement `lookup_draft(context, max_ngram, min_ngram, k)`: for `n` from `max_ngram` down to `min_ngram`, find the most recent earlier occurrence of the last `n` tokens and return up to `k` tokens that followed it; empty if none (prompt lookup decoding, Saxena, 2023; the `ngram` method in vLLM). Numeric. Mutants: first occurrence; matches the suffix with itself; shorter n-gram preferred.

**`speculative-speedup-model`** — Implement `expected_tokens(alpha, k)` `= (1 - alpha**(k+1)) / (1 - alpha)`, `speedup(alpha, k, c)` with draft-to-target cost ratio `c` (Leviathan et al., 2023, Theorem 3.8), and `best_k(alpha, c, k_max)`. Numeric, including `alpha = 1`. Mutants: geometric series without the bonus token; cost `k * c` without the target call; division by zero at `alpha = 1`.

**`speculative-sampling-block`** — Implement `verify_block(p, q, draft_tokens, u_accept, u_resample) -> (n_accepted, next_token)` with `p` of shape `(K + 1, V)`, `q` of shape `(K, V)`: accept drafts in order with probability `min(1, p/q)`, at the first rejection sample from the normalized residual `max(p - q, 0)`, and after `K` acceptances sample from `p[K]` (Leviathan et al., 2023, Algorithm 1). Statistical judge on the first emitted token and on the token following an accepted prefix. Mutants: continues after a rejection; bonus token sampled from the residual; one uniform reused for acceptance and resampling.

**`speculative-decode-loop`** — Implement `speculative_greedy(target, draft, prompt, k, max_new_tokens, eos_id)`: the draft proposes `k` tokens through its cache, the target scores all of them in one call, accepted tokens are kept, and both caches are truncated to the accepted length before the next round. Output must equal the target's own greedy decode. Cost: one target call per round, target `tokens_processed <= P + rounds * (k + 1)`, draft `tokens_processed <= P + rounds * (k + 1)`; rebuilding either cache from scratch exceeds both, and on the high-agreement fixture the target must make fewer than `g / 2` calls. Mutants: cache not rolled back after rejection; draft continues from a rejected token; EOS inside an accepted run ignored; `max_new_tokens` overshoot. [kit: `draft_target_pair`, `KVCache.truncate`]

**`dynamic-draft-length`** — Implement `next_k(k, n_accepted, n_drafted, cfg)` for the two specified policies: the additive heuristic of Hugging Face assisted generation (all accepted: `k + 2`; otherwise `max(1, k - 1)`), and confidence stopping (`should_stop_drafting(draft_prob, threshold)`); then `simulate(policy, acceptance_trace, costs)` returning total cost. Cost judge: on fixture traces the adaptive policy must beat every fixed `k` by the stated margin. Mutants: grows on any acceptance; never shrinks below the initial `k`; confidence compared on logits.

**`typical-acceptance`** — Implement `typical_accept(p, draft_tokens, epsilon, delta)`: accept draft token `x` while `p[x] >= min(epsilon, delta * exp(-H(p)))`, stopping at the first failure, with the first token always accepted greedily (Medusa, Cai et al., 2024; the entropy-gated family that includes vLLM's lossy verification). Numeric. Mutants: entropy in bits; `max` in place of `min`; continues after a failure. The statement says plainly that this is not distribution-preserving.

**`batched-speculative-accept`** — Implement `advance_batch(tokens, lengths, drafts, n_accepted, next_tokens)`: rows accept different numbers of draft tokens; return the updated padded token matrix, new lengths, and per-row cache lengths to truncate to. Property: each row equals its own single-sequence result. Mutants: all rows truncated to the batch minimum (correct but wasteful, rejected by a count check); padding written inside accepted spans; lengths off by the bonus token.

**`mtp-head-draft`** — Implement `mtp_draft(hidden, heads, w_out, k)`: head `i` maps the last hidden state to the hidden state predicting token `t + 1 + i` (a residual MLP as specified), tokens are the argmax through the shared output head; then verify with kit `ops` (Medusa-style heads; multi-token prediction modules as shipped with DeepSeek-V3 and served by vLLM's `mtp` method). Numeric. Mutants: every head reads the previous head's output when the spec is parallel heads; head index off by one; separate unembedding assumed.

**`tree-attention-mask`** — Implement `tree_inputs(parents, prefix_len) -> (mask, position_ids)` for a draft token tree in topological order (each node attends to the prefix and to its ancestors only; position is prefix length plus depth), and `verify_tree(parents, tokens, target_argmax)` returning the longest accepted root path (SpecInfer, Miao et al., 2023; EAGLE-2, Li et al., 2024). Property: each node's mask row equals the mask of its path as a linear sequence. Mutants: siblings visible; positions by flat index; path chosen by depth-first order rather than acceptance.

**`multi-draft-rejection`** — Implement `multi_draft_step(p, q, candidates, u_accept, u_resample)` for `n` i.i.d. candidates drawn from `q`: recursive rejection sampling, accepting candidate `i` with `min(1, p_i / q)` and replacing `p` by the normalized residual after each rejection, finally sampling from the last residual (SpecInfer, Miao et al., 2023; analyzed in SpecTr, Sun et al., 2023). Statistical judge: output follows `p` for any `n`. Mutants: residual not updated between candidates; first accepted of independent tests against the original `p`; residual left unnormalized.

**`early-exit-self-speculation`** — Implement `self_speculative_greedy(lm, prompt, exit_layer, k, max_new_tokens)`: draft tokens with the first `exit_layer` layers through the shared head, then verify by running only the remaining layers on the draft positions, reusing the early layers' hidden states and cache (LayerSkip, Elhoushi et al., 2024). Output equals full-model greedy. Cost: `layer_tokens` under a bound that recomputing the early layers during verification exceeds. Mutants: early layers recomputed; late-layer cache not rolled back; exit hidden state unnormalized before the head. [kit: `LayeredLM`]

**`suffix-automaton-draft`** — Implement `SuffixIndex` with `extend(token)` (online construction of a suffix automaton over all previously generated outputs and the current prompt) and `draft(context, max_len, min_match)` returning the continuation after the longest suffix match, chosen by occurrence count among candidates (SuffixDecoding, Oliaro et al., 2024; the `suffix` method in vLLM). Speed: amortized O(1) `extend` and O(match) `draft`, within 5x of the reference on 100,000 tokens. Mutants: rescans history per draft; shortest match; counts not propagated along suffix links.

### Track 9: `sizing` Sizing and Benchmarking (order 90)

Predict memory and speed on paper, then measure the right thing.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `param-count` | Counting Parameters | easy | numeric | python | - | M2 |
| 20 | `kv-cache-bytes` | KV Cache Bytes per Token | easy | numeric | python | - | M2 |
| 30 | `latency-metrics` | TTFT, ITL, and TPOT | easy | numeric | python | - | M2 |
| 40 | `latency-percentiles` | Percentiles Done Right | easy | numeric, property | python | - | M2 |
| 50 | `flops-per-token` | FLOPs for Prefill and Decode | medium | numeric | python | `param-count` | M2 |
| 60 | `memory-bandwidth-bound` | The Memory-Bandwidth Bound on Tokens per Second | easy | numeric | python | `param-count`, `kv-cache-bytes` | M2 |
| 70 | `memory-budget-fit` | How Many Sequences Fit | easy | numeric | python | `kv-cache-bytes` | M5 |
| 80 | `goodput-slo` | Goodput Under an SLO | easy | numeric | python | `latency-metrics`, `latency-percentiles` | M5 |
| 90 | `stream-latency-probe` | Measuring a Token Stream | easy | simclock | python | `latency-metrics` | M5 |
| 100 | `capacity-planner` | Capacity Planning from a Cost Model | medium | numeric, property | python | `goodput-slo` | M5 |
| 110 | `real-model-sizing` | Sizing a Real Checkpoint | easy | numeric | python | `param-count`, `kv-cache-bytes`, `safetensors-parse` | M3 |

**`param-count`** — Implement `count_params(cfg) -> dict` for a decoder config (vocabulary, hidden size, layers, heads, KV heads, head dim, MLP type and width, tied or untied head, biases, norm gains), returning per-component and total counts. Worked examples are the two Granite configs. Mutants: tied head counted twice; GQA K and V sized as full heads; SwiGLU counted as two matrices.

**`kv-cache-bytes`** — Implement `kv_bytes_per_token(cfg, dtype)` `= 2 * n_layers * n_kv_heads * head_dim * itemsize` and `kv_bytes(cfg, dtype, batch, context, window_layers)` where sliding-window layers cap at their window. Numeric. Mutants: query heads in place of KV heads; factor of 2 missing; windowed layers charged for the full context.

**`latency-metrics`** — Implement `request_metrics(t_start, token_times)` returning TTFT, the inter-token latency list, TPOT defined as `(t_last - t_first) / (n - 1)` (first token excluded, the AIPerf inter-token definition; GuideLLM's TPOT includes it, and the statement shows both), end-to-end latency, and output tokens per second; `n = 1` gives TPOT `None`. Mutants: first token included; ITL from request start; tokens per second over generation time only when the spec is end-to-end.

**`latency-percentiles`** — Implement `percentile_nearest_rank(values, q)`, `percentile_linear(values, q)` (the numpy default), and `merge_percentile(shards, q)` which must pool raw samples. Property: nearest-rank returns an observation; the pooled p99 differs from the mean of shard p99s on the fixture. Mutants: averaging shard percentiles; rank `floor` instead of `ceil`; unsorted input assumed sorted.

**`flops-per-token`** — Implement `prefill_flops(cfg, t)` and `decode_flops(cfg, context)` with the stated convention (a matmul of `m x n` by `n x p` costs `2 m n p`; norms and activations ignored): projections, attention scores and mixing over the context with GQA, MLP, and one head projection. Numeric. Mutants: decode charged the quadratic prefill term; head projection charged for all positions in decode; attention term using KV heads for the score count.

**`memory-bandwidth-bound`** — Implement `decode_tokens_per_s(weight_bytes, kv_bytes_read, bandwidth_bytes_per_s, batch)` and `crossover_batch(...)` against a stated compute rate: each decode step reads all weights once regardless of batch and each sequence's KV cache once, so single-stream speed is at most bandwidth over bytes touched. Worked example: the 8-bit 3B model on 400 GB/s unified memory. Mutants: weights read once per sequence; KV ignored; GB versus GiB.

**`memory-budget-fit`** — Implement `max_sequences(memory_bytes, weight_bytes, cfg, dtype, context, overhead_fraction)` and `max_context(...)` for a given batch. Numeric; results are floors and never negative. Mutants: overhead applied after the division; rounding up; weights forgotten.

**`goodput-slo`** — Implement `slo_attainment(requests, ttft_slo, tpot_slo)` and `goodput(runs, ttft_slo, tpot_slo, target)` where `runs` maps a request rate to measured requests: the highest rate whose attainment is at least `target` with all lower rates also passing (DistServe, Zhong et al., 2024). Numeric. Mutants: SLO on mean latency; counts requests meeting either SLO; picks the highest passing rate above a failing one.

**`stream-latency-probe`** — Implement `async probe(stream, clock) -> metrics` consuming an async token stream under a simulated clock, recording the request start before the first await, timestamps per token, and handling an empty or failing stream; then `run_closed_loop(make_stream, concurrency, n)`. Simclock judge compares exact virtual timestamps. Mutants: start time taken after the first await; timestamps taken after processing delay; a failed stream counted as a zero-latency success.

**`capacity-planner`** — Implement `plan(cost, workload, slo)` for the kit cost model (prefill seconds per token, decode step `base + per_seq * batch`): the largest decode batch whose TPOT meets the SLO, the resulting tokens per second per replica, and the replica count for a given request rate and mean lengths. Property: one fewer replica violates the SLO in the model. Mutants: prefill time ignored in the utilization; batch chosen by throughput only; ceiling omitted.

**`real-model-sizing`** — Implement `size_report(header, config)` from the parsed safetensors header and `config.json` of each Granite model: total parameters from tensor shapes (deduplicating tied tensors), bytes on disk, KV bytes per token at BF16, and the predicted size of an 8-bit group-64 quantization including scales and biases. Judged against fixture numbers (352.4 M; 3.66 B; the 3.89 GB file within 2 percent). Mutants: tied embedding double counted; quantization overhead ignored; KV sized with 16 or 40 heads. [kit: `assets`]

### Track 10: `mlx` MLX GPU Programming (order 100)

All problems: frameworks `["mlx"]`, `requires = "mlx-gpu"`, milestone M3. The oracle is
`reference/mlx.py` [kit: MLX-only oracle]. API names are those of MLX 0.32.4 and
mlx-lm 0.32.2 (`docs/VERSIONS.md`); re-check signatures when the pin moves. Speed cases
compare with the MLX reference only.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `mlx-lazy-eval-placement` | Where to Evaluate | easy | numeric, cost | mlx | `cached-greedy-decode` | M3 |
| 20 | `mlx-mask-free-indexing` | Filtering Without Boolean Indexing | easy | numeric | mlx | `top-k-filtering` | M3 |
| 30 | `mlx-dtype-discipline` | Staying in Half Precision | easy | numeric, cost | mlx | - | M3 |
| 40 | `mlx-memory-accounting` | Measuring Unified Memory | easy | numeric, cost | mlx | - | M3 |
| 50 | `mlx-fast-attention` | Using the Fused Attention Kernel | easy | numeric | mlx | `causal-self-attention` | M3 |
| 60 | `mlx-weights-load` | Loading Weights into a Module Tree | easy | numeric | mlx | `safetensors-parse` | M3 |
| 70 | `mlx-compile-step` | Compiling a Decode Step | medium | numeric, cost | mlx | `mlx-lazy-eval-placement` | M3 |
| 80 | `mlx-compile-shapes` | Compiling Without Retracing | medium | numeric, cost | mlx | `mlx-compile-step` | M3 |
| 90 | `mlx-preallocated-kv-cache` | A Step-Allocated KV Cache | medium | numeric, cost | mlx | `preallocated-kv-cache` | M3 |
| 100 | `mlx-quantized-linear` | A Quantized Linear Layer | medium | numeric, cost | mlx | `groupwise-4bit`, `quantized-matmul` | M3 |
| 110 | `mlx-quantize-module-tree` | Quantizing a Model with a Mixed Recipe | medium | numeric, cost | mlx | `mlx-quantized-linear` | M3 |
| 120 | `mlx-streaming-generate` | Streaming Generation with Async Evaluation | medium | numeric, cost | mlx | `mlx-lazy-eval-placement` | M3 |
| 130 | `mlx-benchmark-harness` | Benchmarking a Lazy Framework | medium | numeric, cost | mlx | `mlx-lazy-eval-placement`, `latency-metrics` | M3 |
| 140 | `mlx-metal-kernel-elementwise` | A Custom Metal Kernel | medium | numeric | mlx | - | M3 |
| 150 | `mlx-metal-kernel-dequant` | A Metal Kernel That Unpacks and Dequantizes | hard | numeric, speed | mlx | `mlx-metal-kernel-elementwise`, `bit-packing` | M3 |
| 160 | `mlx-batched-generate` | Batched Generation with Ragged Prompts | hard | numeric, cost | mlx | `batched-cached-decode`, `padded-batch-invariance` | M3 |
| 170 | `mlx-real-model-forward` | A Real Model in mlx.nn | hard | numeric | mlx | `real-model-logits`, `mlx-weights-load` | M3 |
| 180 | `mlx-real-8bit-generate` | Running a 3B Model at 8 Bits | hard | numeric, cost | mlx | `mlx-real-model-forward`, `mlx-quantized-linear`, `mlx-streaming-generate` | M3 |

**`mlx-lazy-eval-placement`** — Implement `generate(model, prompt, n)` over an `MLXTinyLM`: build each step lazily and force evaluation exactly once per generated token, evaluating the token and the cache state together so the graph never grows across steps. Cost: `spy.evals == n`, `spy.implicit == 0`. Mutants: `.item()` on every intermediate; no evaluation until the end (graph grows; a peak-memory limit fails it); cache arrays left unevaluated. [kit: `EvalSpy`, `MLXTinyLM`]

**`mlx-mask-free-indexing`** — Implement `top_k_filter(logits, k)` and `scatter_penalty(logits, ids, value)` in MLX, which has no boolean-mask indexing or in-place scatter with duplicates: use `mx.where`, `mx.take_along_axis`, `mx.put_along_axis`, `mx.argpartition`. Numeric against a numpy oracle. Mutants: converts to numpy and back (implicit evaluation, caught by `EvalSpy`); duplicate ids applied twice; `argpartition` order assumed sorted.

**`mlx-dtype-discipline`** — Implement `forward_f16(params, x)` for a small MLP whose parameters are float16: the result must be float16 and no intermediate may be float32 (Python scalars are weakly typed, but `mx.array(0.5)` and numpy constants are float32 and promote everything). Cost: peak memory under a limit a float32 pipeline exceeds, plus a dtype check. Mutants: float32 constant in the graph; final `astype` hiding the promotion; epsilon that underflows to 0 in float16. [kit: `mlx_memory`]

**`mlx-memory-accounting`** — Implement `measure(fn, *args) -> dict` returning the peak and active bytes attributable to `fn` using `mx.reset_peak_memory`, `mx.get_peak_memory`, `mx.get_active_memory`, `mx.clear_cache`, and `predicted_bytes(shapes, dtype)`; the result of `fn` must be evaluated inside the measurement. Numeric with a stated 10 percent band on fixtures whose sizes dominate overhead. Mutants: lazy result never evaluated (peak near zero); cache memory counted as active; peak not reset.

**`mlx-fast-attention`** — Implement `attend(q, k, v, n_cached, causal)` with `mx.fast.scaled_dot_product_attention`: `mask="causal"` for a square prefill, an explicit additive mask when new tokens attend to a cache (the string form is wrong there), and no mask for a single-token step. Numeric against a manual implementation. Mutants: `"causal"` with a non-square cache layout; boolean mask inverted; `scale` omitted.

**`mlx-weights-load`** — Implement `load_into(module, path)`: read a safetensors file with `mx.load` (lazy), map checkpoint names to the module tree with `tree_unflatten`, cast BF16 to the module's dtype, and raise on missing or unexpected keys (`strict`), without evaluating tensors that are not used. Numeric on outputs; cost on evaluations. Mutants: silent partial load; transposed Linear weights from a `(in, out)` checkpoint variant; everything evaluated at load. [kit: `EvalSpy`]

**`mlx-compile-step`** — Implement `make_step(model)` returning a function compiled with `mx.compile` that maps `(token, cache_state)` to `(next_token, cache_state)` with fixed shapes. Cost: over 50 steps the Python body is traced once (`TraceCounter`), and results equal the uncompiled reference. Mutants: compiles inside the loop (retrace every step); Python-side branch on an array value; captures a stale array as a constant. [kit: `TraceCounter`, `MLXTinyLM`]

**`mlx-compile-shapes`** — Implement `make_sampler_step(state)` where prompts of varying length and a mutable random or penalty state are involved: declare captured state with `inputs=`/`outputs=`, and bound retracing across lengths either with `shapeless=True` where the body allows it or by padding to buckets. Cost: at most 3 traces across 40 calls with 12 distinct lengths; state updates visible after each call. Mutants: state captured as a constant (never updates); a retrace per length; `shapeless=True` on a body that bakes in a shape (wrong numbers).

**`mlx-preallocated-kv-cache`** — Implement `class KVCache` with `update_and_fetch(keys, values)` and `offset`, growing the buffers in steps of 256 tokens and writing new entries by slice assignment (the mlx-lm `KVCache` design). Cost: the buffer's shape changes at most `ceil(n / 256)` times over `n` appended tokens and peak memory stays under a limit that concatenation per step exceeds. Mutants: concatenate every step; returns the whole buffer including the unused tail; growth by exactly the needed amount.

**`mlx-quantized-linear`** — Implement `class QLinear` built from a float `nn.Linear` with `mx.quantize(w, group_size, bits)` and evaluated with `mx.quantized_matmul`, matching `nn.QuantizedLinear`, including `mode="mxfp4"` where `mx.quantize` returns two arrays. Cost: parameter bytes at 4 bits are under 0.2 of the float32 layer. Mutants: stores the dequantized weight; `transpose` flag wrong; bias dropped. Forbid `mlx.nn.QuantizedLinear`.

**`mlx-quantize-module-tree`** — Implement `quantize_model(model, recipe)` walking the module tree and replacing each `nn.Linear` and `nn.Embedding` according to `recipe(path, module) -> {bits, group_size} | None` (the `class_predicate` mechanism of `mlx.nn.quantize`), then `bits_per_weight(model)`. Numeric on outputs against the oracle; cost on total parameter bytes for a mixed 4/8-bit recipe. Mutants: predicate result ignored for embeddings; children of lists skipped; bits per weight ignoring scales and biases.

**`mlx-streaming-generate`** — Implement `stream_generate(model, prompt, n, detok)` as a generator: schedule step `t + 1` with `mx.async_eval` before reading token `t` with `.item()`, so the GPU works while Python detokenizes (the mlx-lm `generate_step` pattern). Cost: `EvalSpy` shows each `async_eval` of step `t + 1` precedes the read of step `t`, and exactly one read per token. Mutants: synchronous `mx.eval` then read; async evaluation of the token but not the cache; an extra speculative step after the final token. [kit: `EvalSpy`, `MLXTinyLM`]

**`mlx-benchmark-harness`** — Implement `benchmark(step, n_warmup, n_runs, clock)` returning prompt tokens per second and generation tokens per second: warm up (compilation and allocation), evaluate inside the timed region, call `mx.synchronize` before reading `clock`, and report the median. `clock` and `step` are kit fakes whose work completes only on evaluation, so timing a lazy graph measures zero. Mutants: no evaluation inside the timer; warm-up runs included; mean in place of median. [kit: `EvalSpy`]

**`mlx-metal-kernel-elementwise`** — Implement `fused_silu_mul(a, b)` with `mx.fast.metal_kernel`: one kernel computing `silu(a) * b` per element (`input_names`, `output_names`, `source`, grid equal to the element count, templated on dtype for float16 and float32). Numeric against MLX ops, on shapes that are not row-contiguous. Mutants: grid of the first dimension only; float32 hard-coded; `elem` indexing wrong for transposed input with `ensure_row_contiguous=False`.

**`mlx-metal-kernel-dequant`** — Implement `dequant4(words, scales, biases, group_size, shape)` as a Metal kernel: each thread unpacks one 4-bit code from the packed uint32 words and writes `code * scale + bias` for its group. Numeric against `mx.dequantize`; speed within 3x of `mx.dequantize` on a `4096 x 4096` layer, where a Python-loop or per-group formulation is far slower. Mutants: nibble order reversed; group index computed from the flat index without the row; threadgroup size that skips the tail.

**`mlx-batched-generate`** — Implement `batch_generate(model, prompts, max_new_tokens, eos_id)` for prompts of different lengths: left-pad, build the additive mask and per-row position offsets, keep one batched cache, stop each row at EOS, and finish when all rows are done. Each row must equal its single-sequence greedy result. Cost: one evaluation per step and no step after all rows finish. Mutants: pads attended; positions from the padded start; finished rows keep changing the output. [kit: `MLXTinyLM`, `EvalSpy`]

**`mlx-real-model-forward`** — Implement `class Granite(nn.Module)` for `granite-4.0-350m` in `mlx.nn`, loadable from the original safetensors names, using `mx.fast.rms_norm`, `mx.fast.rope` or `nn.RoPE`, and `mx.fast.scaled_dot_product_attention`. Numeric against fixture logits (`rtol` 1e-3 in float32; a second case in float16 with a stated looser tolerance) and exact argmax tokens. Mutants: attention scale `1/sqrt(d)`; RoPE base 10000; embedding multiplier missing. [kit: `assets`]

**`mlx-real-8bit-generate`** — Implement `load_and_generate(path, prompt_ids, max_new_tokens)` for `lmstudio-community/granite-4.2-3b-MLX-8bit`: build the 40-layer model with quantized linear layers directly from the packed weights (never materializing float weights), then generate with a KV cache and async evaluation. Judged on exact fixture token ids and on peak memory under 5 GB (a dequantized float16 load is over 7 GB). Mutants: dequantize at load; untied head assumed tied; thinking-template tokens mishandled in the prompt. Time limit 120 s; the case is skipped when the asset is absent. [kit: `assets`, `mlx_memory`]

### Track 11: `cuda` CUDA GPU Programming (order 110)

All problems: frameworks `["torch"]`, `requires = "cuda"`, milestone M4. The owner has no
NVIDIA GPU, so every problem is split into **CPU logic cases**, which run everywhere and
decide validation, and **GPU cases** marked `@case(..., requires="cuda")`, which are
skipped without CUDA [kit: per-case requirement]. Every solution takes `device` as an
argument and must work for `torch.device("cpu")`. "CPU:" and "GPU:" below say what each
half checks.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `cuda-inference-mode` | Inference Mode | easy | numeric, property | torch | - | M4 |
| 20 | `cuda-device-placement` | Keeping Tensors on One Device | easy | numeric, property | torch | - | M4 |
| 30 | `cuda-timing-sync` | Timing Asynchronous Work | easy | numeric, cost | torch | `latency-metrics` | M4 |
| 40 | `cuda-sdpa-attention` | Fused Attention with the Right Mask | easy | numeric | torch | `causal-self-attention` | M4 |
| 50 | `cuda-sync-free-decode-loop` | A Decode Loop Without Host Syncs | medium | numeric, cost | torch | `cached-greedy-decode`, `categorical-sample` | M4 |
| 60 | `cuda-autocast-inference` | Mixed-Precision Inference | medium | numeric, property | torch | `stable-softmax` | M4 |
| 70 | `cuda-pinned-transfer` | Pinned Memory and Non-Blocking Transfers | medium | numeric, cost | torch | `cuda-device-placement` | M4 |
| 80 | `cuda-memory-accounting` | Reading the CUDA Allocator | easy | numeric | torch | `kv-cache-bytes` | M4 |
| 90 | `cuda-oom-backoff` | Recovering from Out-of-Memory | medium | numeric, cost | torch | - | M4 |
| 100 | `cuda-static-kv-cache` | A Static KV Cache | medium | numeric, property | torch | `preallocated-kv-cache` | M4 |
| 110 | `cuda-compile-once` | Compiling a Decode Step Once | hard | numeric, cost | torch | `cuda-static-kv-cache` | M4 |
| 120 | `cuda-graph-replay` | Capturing a CUDA Graph | hard | numeric, cost | torch | `cuda-static-kv-cache` | M4 |
| 130 | `cuda-triton-rmsnorm` | A Triton Kernel for RMSNorm | medium | numeric, property | torch | `rmsnorm` | M4 |
| 140 | `cuda-triton-dequant-linear` | A Fused Dequantize-and-Multiply Kernel | hard | numeric, property | torch | `cuda-triton-rmsnorm`, `quantized-matmul` | M4 |

**`cuda-inference-mode`** — Implement `predict(model, x, device)` and `as_inference(fn)`: run under `torch.inference_mode()` with the module in eval mode, restoring the previous training flag afterward, returning tensors with no `grad_fn`. CPU: outputs have `requires_grad == False`, dropout is inactive, the mode is restored even on exception. GPU: peak `max_memory_allocated` is below the autograd-retaining mutant's. Mutants: `no_grad` forgotten; `eval()` not restored; mode left on after an exception.

**`cuda-device-placement`** — Implement `run(model, inputs, device)` where `inputs` is a nested structure and the model creates masks and position ids internally: every created tensor must use the device and dtype of its inputs (`torch.arange(..., device=x.device)`), and outputs are returned on `device`. CPU: a kit meta-device check wraps tensor factories and fails any creation without an explicit device inside the solution; numeric equality. GPU: no CPU tensors appear in the forward pass. Mutants: `torch.arange` on the default device; `.cpu()` round trip in the middle; inputs moved one leaf short in nested dicts.

**`cuda-timing-sync`** — Implement `time_region(fn, device_api, n)` returning median seconds per call, where `device_api` provides `sync()` and `clock()`: synchronize before starting and before stopping the timer, and exclude warm-up. CPU: the kit `FakeDevice` completes work only on `sync`, so exact virtual times are checked. GPU: the same function with `torch.cuda.synchronize` and `perf_counter` must report at least the time measured by `torch.cuda.Event`. Mutants: no sync before stop; sync inside but not before start; first call included. [kit: `FakeDevice`]

**`cuda-sdpa-attention`** — Implement `attend(q, k, v, n_cached, key_padding)` with `torch.nn.functional.scaled_dot_product_attention`: `is_causal=True` only for a square prefill, a boolean mask (True means attend) otherwise, combined correctly with key padding, in GQA layout. CPU: numeric against a manual implementation, including the cache case. GPU: same numbers in float16, and speed within 2x of the reference. Mutants: `is_causal` with a cache; boolean mask polarity inverted; `attn_mask` and `is_causal` both set.

**`cuda-sync-free-decode-loop`** — Implement `decode(model, prompt, n, uniforms, device)`: sample on the device from supplied uniforms, keep an `(n,)` token buffer on the device, track EOS with a tensor flag checked every `check_every` steps, and transfer results once at the end. CPU: `SyncSpy` allows `ceil(n / check_every) + 1` host synchronizations and exact tokens are checked. GPU: speed within 1.5x of the reference where the per-token `.item()` mutant is slower than that. Mutants: `.item()` per token; Python-side EOS check each step; token list built on the host. [kit: `SyncSpy`]

**`cuda-autocast-inference`** — Implement `forward_amp(model, x, device, dtype)` using `torch.autocast(device_type=device.type, dtype=dtype)` for the body while computing the final log-softmax and sampling statistics in float32. CPU: runs with `dtype=torch.bfloat16`; checks that hidden activations are bfloat16, final probabilities float32, and the result is within a stated tolerance of full float32. GPU: float16, with a logit range that overflows a float16 softmax. Mutants: whole pipeline in half precision (inf logits); probabilities returned as bfloat16; `device_type` hard-coded to `"cuda"`.

**`cuda-pinned-transfer`** — Implement `class Prefetcher(batches, device, to_device)` yielding batches while the next one is already being transferred: allocate host buffers through the supplied `pin` function and move them with `non_blocking=True`. CPU: `pin` and `to_device` are kit spies; the test checks that transfer `i + 1` starts before batch `i` is consumed, every tensor was pinned first, and order is preserved. GPU: tensors report `is_pinned()` before transfer and total time is within 1.5x of the reference. Mutants: transfer on demand; pageable memory with `non_blocking=True`; last batch dropped.

**`cuda-memory-accounting`** — Implement `predicted_allocated(shapes, dtype)` (each allocation rounded up to 512 bytes), `fragmentation(stats)` from a `torch.cuda.memory_stats()`-shaped dict (`reserved - allocated`, and the inactive split fraction), and `kv_budget(stats, total_bytes, safety)`. CPU: numeric on fixture dicts captured from a real GPU and committed with the problem. GPU: the prediction is within 2 percent of `torch.cuda.max_memory_allocated()` for the fixture allocations. Mutants: no block rounding; reserved treated as in use; safety margin applied to total instead of free.

**`cuda-oom-backoff`** — Implement `run_batched(model, items, batch_size)`: on `torch.OutOfMemoryError`, release references, call the supplied `empty_cache`, halve the batch size, and retry that slice; keep the reduced size for later slices; raise if a single item fails. CPU: the kit `OOMModel` raises above a threshold; the test checks results in order, the exact sequence of batch sizes tried, and that `empty_cache` ran between attempts. GPU: a real allocation-heavy model completes without a crash. Mutants: retries the same size forever; restarts from the first item (duplicates); catches every `RuntimeError`. [kit: `OOMModel`]

**`cuda-static-kv-cache`** — Implement `StaticCache(n_layers, batch, n_kv_heads, max_len, head_dim, device, dtype)` with `update(layer, k, v, pos)` using `index_copy_` on preallocated buffers and `mask(pos, n_new)` so attention ignores the unused tail; every tensor the model sees has a shape independent of the step. CPU: numeric equality with a growing cache, constant shapes and constant `data_ptr()` across steps, NaN-filled tail. GPU: no growth of `memory_reserved` across 256 steps. Mutants: `torch.cat`; mask covering the tail off by one; buffers reallocated when the batch finishes a row.

**`cuda-compile-once`** — Implement `make_step(model, cache)` returning `torch.compile(step, ...)` such that 64 decode steps and prompts padded to a multiple of 16 compile a bounded number of graphs. CPU: compiled with `backend="eager"` (no compiler needed); the test reads the Dynamo frame-compile counters and requires at most 2 compilations and no graph breaks, with outputs equal to eager. GPU: `mode="reduce-overhead"`, speed within 1.5x of the reference. Mutants: Python int position baked into guards (recompiles each step); data-dependent Python branch (graph break); shapes that grow. Authoring note: confirm the counter API against torch 2.15 before relying on it.

**`cuda-graph-replay`** — Implement `GraphedStep(step_fn, example_inputs, graph)`: warm up, capture `step_fn` on static placeholder tensors, and on each call copy new inputs into the placeholders with `copy_`, `replay()`, and clone the static outputs. CPU: `graph` is the kit `FakeCudaGraph`, which recomputes from the captured storages, so rebinding a placeholder yields stale output and missing clones alias across calls. GPU: `torch.cuda.CUDAGraph`, results equal to eager and speed within 1.5x of the reference. Mutants: placeholder rebound instead of copied into; outputs returned without clone; capture without warm-up. [kit: `FakeCudaGraph`]

**`cuda-triton-rmsnorm`** — Write `rmsnorm_kernel` in the Triton language subset: one program per row, loading the row in blocks of `BLOCK` with a tail mask, accumulating the sum of squares, then a second pass writing `x / rms * weight`. CPU: the kernel runs under the kit `tritonshim` and is compared with the oracle on row lengths that are not a multiple of the block. GPU: the same source compiled by Triton, numeric and within 2x of `torch` RMSNorm. Mutants: no tail mask (reads garbage); `other=1.0` on masked loads polluting the sum; eps outside the root. [kit: `tritonshim`]

**`cuda-triton-dequant-linear`** — Write `qlinear_kernel`: for each output row and each block of inputs, load packed 4-bit codes, unpack with shifts and masks, apply the group's scale and bias, and accumulate the dot product with `x`, never writing the dequantized weights to memory. CPU: under `tritonshim`, numeric against `quantized-matmul`'s oracle and a property over group sizes and ragged tails. GPU: within 3x of a dequantize-then-`matmul` baseline while allocating no weight-sized buffer. Mutants: nibble order; group index from the block instead of the column; accumulator in float16. [kit: `tritonshim`]

### Track 12: `serving` Serving Systems (order 120)

Schedulers, queues, and endpoints: what happens between a request arriving and its last token leaving.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `dynamic-batcher` | Dynamic Batching Policy | medium | simclock | python | - | M1 |
| 20 | `sse-token-stream` | Streaming Tokens over Server-Sent Events | medium | service | python | - | M1 |
| 30 | `token-bucket-limiter` | Token-Based Rate Limiting | easy | simclock | python | - | M2 |
| 40 | `admission-control` | Admission Control and Backpressure | easy | simclock | python | - | M2 |
| 50 | `metrics-endpoint` | A Metrics Endpoint | easy | service | python | `latency-metrics` | M2 |
| 60 | `continuous-batching-scheduler` | Continuous Batching | hard | simclock | python | `dynamic-batcher` | M2 |
| 70 | `priority-scheduler` | Priorities Without Starvation | medium | simclock | python | `continuous-batching-scheduler` | M2 |
| 80 | `deadline-cancellation` | Deadlines and Cancellation | medium | simclock | python | `continuous-batching-scheduler` | M2 |
| 90 | `chunked-prefill-scheduling` | Chunked Prefill Under a Token Budget | medium | simclock | python | `continuous-batching-scheduler` | M2 |
| 100 | `vtc-fair-scheduler` | Fair Scheduling with Virtual Token Counters | medium | simclock | python | `continuous-batching-scheduler` | M2 |
| 110 | `openai-chat-endpoint` | An OpenAI-Compatible Chat Endpoint | medium | service | python | `sse-token-stream` | M2 |
| 120 | `least-loaded-balancer` | Power-of-Two-Choices Load Balancing | easy | simclock, statistical | python | - | M5 |
| 130 | `retry-backoff-client` | A Client That Backs Off | easy | simclock | python | `admission-control` | M5 |
| 140 | `graceful-drain` | Graceful Drain | medium | simclock | python | `deadline-cancellation` | M5 |
| 150 | `streaming-backpressure` | Slow Consumers | medium | simclock | python | `continuous-batching-scheduler` | M5 |
| 160 | `open-loop-load-generator` | An Open-Loop Load Generator | medium | simclock, statistical | python | `latency-metrics` | M5 |
| 170 | `slo-deadline-scheduler` | Earliest-Deadline-First with Shedding | medium | simclock | python | `priority-scheduler` | M5 |
| 180 | `multi-lora-batching` | Batching Requests Across Adapters | medium | numeric, speed | t/j/m | `lora-merge` | M5 |
| 190 | `prefix-aware-router` | Prefix-Cache-Aware Routing | medium | simclock | python | `prefix-cache-reuse`, `least-loaded-balancer` | M5 |
| 200 | `autoscaler-policy` | An Autoscaling Policy | medium | simclock | python | `capacity-planner` | M5 |
| 210 | `preemption-recompute` | Preemption by Recompute | hard | simclock | python | `continuous-batching-scheduler`, `paged-kv-allocator` | M5 |
| 220 | `prefill-decode-disaggregation` | Disaggregated Prefill and Decode | hard | simclock | python | `chunked-prefill-scheduling` | M5 |

All simclock problems count time in integer milliseconds, specify the order of simultaneous events, and include a tie case. [kit: `SimEngine`, `SimRequest`, `workloads`]

**`token-bucket-limiter`** — Implement `TokenBucket(rate_per_s, burst)` with `try_acquire(n) -> bool`, `async acquire(n)` (waits in FIFO order, no barging by small requests), and `refund(n)` for the unused part of a `max_tokens` reservation, charged in LLM tokens. Simclock judge on exact grant times. Mutants: refill computed only on a timer tick; a small request overtakes a waiting large one; balance exceeds `burst`; `n > burst` waits forever instead of raising.

**`admission-control`** — Implement `Admission(max_queued_requests, max_queued_tokens)` with `admit(req) -> Admitted | Rejected(retry_after_s)`, `start(req)`, `finish(req)`: reject when either bound would be exceeded (vLLM `--max-num-queued-reqs`, `--max-num-queued-tokens`), with `retry_after_s` computed from the stated drain-rate estimate. Simclock: queue depth never exceeds the bounds and accepted latency stays under a stated limit under 3x overload. Mutants: unbounded queue; bound checked after enqueue; tokens of running requests counted as queued.

**`metrics-endpoint`** — Implement `make_app(lm)` exposing `POST /generate` and `GET /metrics` in Prometheus text format: counters for requests by status and for prompt and generation tokens, a gauge for running requests, and a TTFT histogram with given buckets (cumulative `le` buckets, `_sum`, `_count`), names following vLLM's `vllm:` conventions as listed in the statement. Service judge parses the exposition after a scripted workload. Mutants: non-cumulative buckets; gauge never decremented on error; `+Inf` bucket missing. [kit: `ScriptedLM`]

**`continuous-batching-scheduler`** — Implement `async run(engine, queue, max_running)`: an iteration-level loop that, each step, admits waiting requests into free slots (FCFS), sends admitted prompts as prefill and all running sequences as decode in one `engine.step`, and retires finished sequences immediately so their slots are reused next step (Orca, Yu et al., 2022; the vLLM engine loop). Simclock: the trace must equal the oracle's exactly, and p50 latency must beat the static-batch mutant's by the stated margin. Mutants: waits for the whole batch to finish; a finished slot idles one step; arrivals during a step lost; a sequence decoded in the same step as its prefill when the spec says the next step.

**`priority-scheduler`** — Extend the waiting queue with priorities (lower value first, FCFS within a level, as vLLM's `priority` policy) and aging: a request's effective priority improves by one level per `aging_ms` waited. Simclock: exact admission order on fixtures, and a bound on the longest wait of a low-priority request under a steady high-priority stream. Mutants: no aging (starvation); ties broken by request id rather than arrival; priority read once at enqueue.

**`deadline-cancellation`** — Implement `submit(req) -> handle` with `handle.cancel()` and per-request deadlines on the continuous-batching loop: a canceled or expired request is removed before the next engine step whether waiting or running, its slot is reused, and its client receives `Canceled` or `DeadlineExceeded`. Simclock: `engine` records no token for a request after its cancellation instant, and total engine work matches the oracle. Mutants: cancellation only honored while waiting; slot leaked; expiry checked only at admission.

**`chunked-prefill-scheduling`** — Implement the step planner `plan(running, prefilling, waiting, max_batch_tokens, long_prefill_threshold)`: decodes are scheduled first (one token each), the remaining budget goes to prefill chunks in FCFS order, and a prompt longer than the remaining budget is split across steps (vLLM chunked prefill; Sarathi-Serve, Agrawal et al., 2024). Simclock: maximum inter-token gap of running requests stays under a stated bound when a 4,000-token prompt arrives. Mutants: prefill first; whole prompt in one step; a partially prefilled request decoded early.

**`vtc-fair-scheduler`** — Implement the Virtual Token Counter scheduler: per-tenant counters charged `w_in * input + w_out * output` tokens as they are served, the next admitted request belongs to the backlogged tenant with the smallest counter, and a tenant that becomes active is lifted to the minimum counter of active tenants (Sheng et al., OSDI 2024, Algorithm 2). Simclock: service difference between two backlogged tenants stays within the paper's bound; an idle tenant cannot bank credit. Mutants: no counter lift; charges at admission using `max_tokens`; counts requests instead of tokens.

**`openai-chat-endpoint`** — Implement `make_app(lm)` serving `POST /v1/chat/completions`: non-streaming response with `choices[0].message`, `finish_reason` (`stop` or `length`), and `usage`; streaming with `chat.completion.chunk` objects (role in the first delta, content deltas, a final chunk with `finish_reason`, optional usage chunk under `stream_options.include_usage`, then `data: [DONE]`); 400 with an OpenAI-shaped error body for invalid input. Service judge. Mutants: `finish_reason` always `stop`; usage counted in characters; missing `[DONE]`; 500 on a missing `messages` field. [kit: `ScriptedLM`]

**`least-loaded-balancer`** — Implement `pick(replicas, load, u1, u2)`: choose two distinct replicas from the supplied uniforms and route to the one with fewer outstanding tokens, ties to the lower id (power of two choices, Mitzenmacher, 2001). Simclock plus statistical: maximum load under the seeded workload stays under a bound that uniform random routing exceeds; choice frequencies follow the exact distribution. Mutants: the same replica drawn twice; compares request counts; ties to the first drawn.

**`retry-backoff-client`** — Implement `async call_with_retry(send, policy, uniforms)`: retry on 429 and 503 only, honoring `Retry-After` when present, otherwise exponential backoff with full jitter from the supplied uniforms, capped delay, bounded attempts, and a total deadline. Simclock on exact send times. Mutants: retries 400s; no jitter (synchronized retries, shown by a thundering-herd case); `Retry-After` ignored.

**`graceful-drain`** — Implement `Server.drain(grace_ms)`: stop admitting (new requests get `Unavailable`), let waiting and running requests finish, cancel what remains at the grace deadline with a final error event, and resolve when the engine is idle. Simclock on the exact outcome of each request. Mutants: waiting requests dropped at once; drain resolves while streams are open; new requests still admitted during drain.

**`streaming-backpressure`** — Implement per-request output buffers between the engine loop and clients: bounded size, tokens coalesced into one chunk when the client is slow, and a client that stays full past `stall_ms` is canceled; the engine loop never awaits a client. Simclock: a stalled client does not change any other request's token times. Mutants: `await queue.put` inside the engine loop; unbounded buffer; tokens dropped when full.

**`open-loop-load-generator`** — Implement `async generate_load(send, rate, n, uniforms)`: schedule arrivals from exponential inter-arrival times computed from the supplied uniforms, independent of response times, and measure latency from the intended send time. Simclock plus statistical (inter-arrival mean within the Hoeffding band on a bounded transform). Mutants: closed loop (waits for the response before the next send); latency from actual send time (coordinated omission); uniform inter-arrivals.

**`slo-deadline-scheduler`** — Implement admission ordered by TTFT deadline (earliest first) with shedding: a waiting request whose deadline cannot be met given the cost model's prefill time is rejected immediately rather than served late. Simclock: SLO attainment on the overload fixture exceeds FCFS's by the stated margin, and no request is served after its deadline. Mutants: FCFS; serves hopeless requests; shedding estimate ignores requests ahead in the queue.

**`multi-lora-batching`** — Implement `batched_lora_forward(x, w, a_stack, b_stack, adapter_ids, alpha)`: one base matmul for the whole batch plus per-row low-rank updates gathered by `adapter_ids` (id `-1` means no adapter), vectorized (S-LoRA, Sheng et al., 2023; multi-LoRA serving in vLLM). Speed: within 5x of the reference at batch 64 with 16 adapters, where a per-row loop or merging weights per request is far slower. Mutants: per-row Python loop; adapter applied to rows with id `-1`; gather on the rank axis.

**`prefix-aware-router`** — Implement `route(prompt_tokens, replicas)`: score each replica by its longest cached prefix match (an approximate per-replica prefix index you maintain from routed requests), route to the best match if its match rate exceeds a threshold and its queue is within `imbalance` of the least loaded, otherwise fall back to least loaded (Ray Serve `PrefixCacheAffinityRouter`; the llm-d prefix-cache scorer; SGLang router). Simclock: cache-hit tokens and p95 TTFT both beat round-robin by stated margins, and no replica exceeds the imbalance bound. Mutants: affinity never broken (hot replica overloads); index not updated after routing; match on text rather than tokens. [kit: `SimCluster`]

**`autoscaler-policy`** — Implement `desired_replicas(metrics, state, cfg) -> (n, state)` evaluated every period: scale up when queue depth per replica or KV utilization exceeds its threshold (the signals llm-d documents: queue length and `kv_cache_usage_perc`), scale down only after a stabilization window, respect cooldown, bounds, and a replica startup delay during which pending replicas count toward capacity. Simclock: on a step-load fixture, time over SLO and replica-seconds are both under stated limits, with at most the stated number of scaling actions. Mutants: no stabilization (flapping); pending replicas ignored (overshoot); scale-down as fast as scale-up. [kit: `SimCluster`]

**`preemption-recompute`** — Extend the continuous-batching loop for an engine with a finite KV block pool: when a decode step cannot allocate, preempt the lowest-priority, most recently admitted running sequence, free its blocks, and put it at the front of the waiting queue to be recomputed from its prompt plus generated tokens (vLLM preemption by recompute); admit new work only above a free-block watermark. Simclock: no `OutOfBlocks` ever reaches the engine, every request completes with the right token count, and total recomputed tokens stay under the oracle's bound. Mutants: preempts the oldest; restarts generation from scratch (tokens emitted twice); no watermark (preemption thrash).

**`prefill-decode-disaggregation`** — Implement a coordinator over a prefill pool and a decode pool: prefill on the least-loaded prefill worker, transfer the KV cache at a cost proportional to prompt tokens, then join a decode worker's batch; short prompts below a threshold stay local to the decode worker (DistServe, Zhong et al., 2024; NVIDIA Dynamo). Simclock: TTFT and TPOT attainment both exceed the colocated baseline on the mixed workload fixture, and the trace shows no decode step delayed by a prefill. Mutants: transfer cost ignored in placement; every request disaggregated (short prompts pay transfer); decode worker chosen before prefill finishes and left idle-reserved. [kit: `SimCluster`]

### Track 13: `systems` Distributed and LLM Application Systems (order 130)

Models that do not fit on one device, and the systems built around a model.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `device-map-solver` | Solving a Device Map | medium | numeric, property | python | `param-count` | M5 |
| 20 | `mmap-lazy-loader` | Lazy Weight Loading with mmap | medium | numeric, cost | python | `safetensors-parse` | M5 |
| 30 | `embedding-pooling` | Pooling and Normalizing Embeddings | easy | numeric, property | t/j/m | `attention-mask-builder` | M5 |
| 40 | `vector-topk-search` | Exact Vector Search | easy | numeric, speed | t/j/m | `embedding-pooling` | M5 |
| 50 | `binary-rescore-search` | Binary Embeddings with Rescoring | medium | numeric, cost | t/j/m | `vector-topk-search` | M5 |
| 60 | `ivf-index-search` | An Inverted-File Index | hard | numeric, cost, property | t/j/m | `vector-topk-search` | M5 |
| 70 | `rerank-batching` | Batching a Reranker | easy | numeric, cost | python | `pad-batch-left` | M5 |
| 80 | `semantic-cache` | A Semantic Response Cache | medium | numeric, simclock | python | `vector-topk-search` | M5 |
| 90 | `cascade-router` | Routing Through a Model Cascade | easy | numeric, cost | python | - | M5 |
| 100 | `tool-call-loop` | A Tool-Calling Loop | medium | numeric, cost | python | `tool-call-stream-parser` | M5 |
| 110 | `tensor-parallel-linear` | Tensor-Parallel Layers on Simulated Devices | medium | numeric, cost | t/j/m | `swiglu-mlp` | M5 |
| 120 | `pipeline-parallel-schedule` | Pipeline Parallelism Across Devices | hard | simclock | python | `device-map-solver` | M5 |

**`device-map-solver`** — Implement `solve_device_map(modules, max_memory, no_split)`: assign modules in order to devices in order (then `"disk"`), never splitting a `no_split` group, keeping tied modules together, and reserving the largest layer's size on the first device for activations (the semantics of `accelerate.infer_auto_device_map`, simplified as stated). Property: budgets respected and the assignment is order-preserving. Mutants: best-fit packing that reorders layers; split block; reservation forgotten.

**`mmap-lazy-loader`** — Implement `LazyCheckpoint(file)` with `keys()`, `get(name) -> ndarray view`, and `load_subset(names)` over a safetensors file opened through the kit's counting file: reading the header must not touch tensor data, and `get` touches only that tensor's pages. Cost: `bytes_read` at most header plus requested tensors plus one page each. Mutants: `file.read()` of everything; a copy per access; offsets misaligned by the header length. [kit: `CountingFile`]

**`embedding-pooling`** — Implement `pool(states, mask, mode)` for `mean` (masked), `last` (last real token, either padding side), and `cls`, followed by L2 normalization, and `truncate_matryoshka(e, dim)` which slices then renormalizes (Kusupati et al., 2022). Property: padding never changes the embedding. Mutants: mean over padded positions; truncate without renormalizing; `last` as `[:, -1]` under right padding.

**`vector-topk-search`** — Implement `search(queries, corpus, k)` returning indices and cosine scores of the top `k` per query, ties to the lowest index, vectorized over queries. Speed: within 5x of the reference for 64 queries against 20,000 vectors, where a Python loop over the corpus is far slower. Mutants: unnormalized dot product; full sort with unstable ties; `k > N` crash.

**`binary-rescore-search`** — Implement `binarize(e)` (sign bits packed to uint8), `hamming_topk(q_bits, corpus_bits, m)`, and `search(q, corpus_bits, corpus_int8, k, oversample)`: retrieve `k * oversample` candidates by Hamming distance, rescore with the float query against int8 vectors, return the top `k`. Cost: float rescoring touches exactly `k * oversample` rows; recall against exact search above a stated level on the seeded corpus. Mutants: rescoring everything; Hamming computed on unpacked floats; ranking by ascending similarity.

**`ivf-index-search`** — Implement `build_ivf(corpus, centroids)` (assign each vector to its nearest centroid) and `search(q, index, k, nprobe)` scanning only the `nprobe` nearest lists (Sivic and Zisserman, 2003; the IVF index of Faiss). Cost: distance computations equal `n_centroids` plus the sizes of probed lists, counted by a kit distance function. Property: `nprobe = n_lists` equals exact search; recall is monotone in `nprobe`. Mutants: probes lists by id order; an empty list crashes; duplicates when `k` exceeds the probed total.

**`rerank-batching`** — Implement `rerank(query, docs, scorer, max_batch, top_n)`: sort candidate pairs by length, batch them to minimize padding, score, restore the original order, and return the `top_n` with stable ties. Cost: `scorer.counters.padded_tokens` under a bound that arrival-order batching exceeds; calls at most `ceil(n / max_batch)`. Mutants: arrival-order batches; scores returned in sorted order; truncation that drops the query. [kit: `CountingScorer`]

**`semantic-cache`** — Implement `SemanticCache(threshold, ttl_s, capacity)` with `get(embedding, now)` (best cosine match at or above the threshold and not expired) and `put(embedding, response, now)` with LRU eviction, keyed additionally by a namespace of model and sampling settings. Numeric and simclock for expiry. Mutants: returns the first match above the threshold instead of the best; cross-namespace hits; expired entries refresh their own TTL on lookup.

**`cascade-router`** — Implement `cascade(prompt, models, thresholds, budget)`: call models from cheapest to most expensive, accept the first answer whose confidence (mean token log-probability, as specified) clears that stage's threshold, stop when the budget would be exceeded, and return the answer with its cost (FrugalGPT, Chen et al., 2023). Cost: model calls and spend match the oracle exactly. Mutants: always calls the largest model for comparison; confidence as the sum of log-probabilities (length bias); budget checked after the call. [kit: `ScriptedChatModel`]

**`tool-call-loop`** — Implement `run_agent(model, tools, messages, max_steps)`: call the model, execute every requested tool call (concurrently when several are requested, results appended in call order with matching ids), return validation errors for malformed arguments to the model as tool results rather than raising, and stop on a final answer or `max_steps`. Cost: model calls equal the script's length; numeric on the final transcript. Mutants: only the first of parallel calls executed; tool exception crashes the loop; ids mismatched; no step limit. [kit: `ScriptedChatModel`]

**`tensor-parallel-linear`** — Implement `tp_mlp(x, shards, devs)` for an MLP split across `n` simulated devices: the first layer split by output columns (no communication), the second by input rows with one all-reduce sum of partial outputs (Megatron-LM, Shoeybi et al., 2019), and `shard_weights(w_up, w_down, n)`. Cost: exactly one all-reduce of output size, each device within its memory budget; numeric against the unsplit layer. Mutants: both layers column-split with an all-gather between them (extra communication); shards overlapping by one column; bias added on every device. [kit: `SimDevices`]

**`pipeline-parallel-schedule`** — Implement `async run_pipeline(stages, requests)`: layers are split into stages on different devices with per-stage service times and a transfer delay; stream decode steps of several sequences through the stages so every stage is busy whenever work is available, preserving per-sequence token order. Simclock: throughput within 5 percent of the bottleneck stage's rate and exact per-token ordering. Mutants: one sequence occupies the whole pipeline (no interleaving); a sequence's next token enters before its previous one leaves; transfers modeled as free.

### Track 14: `frontier` Frontier (order 140)

Techniques that reached production engines in 2025 and 2026. All M5. Re-verify each against its source when authoring: this track ages fastest.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `reasoning-budget-processor` | Enforcing a Thinking-Token Budget | medium | numeric, cost | t/j/m | `constrained-greedy-decode` | M5 |
| 20 | `cross-layer-kv-sharing` | Sharing KV Across Layers | medium | numeric, cost | t/j/m | `cached-greedy-decode` | M5 |
| 30 | `diffusion-unmask-step` | One Step of Masked-Diffusion Decoding | medium | numeric, statistical | t/j/m | `categorical-sample` | M5 |
| 40 | `kv-tiered-offload` | A Tiered KV Store | medium | simclock, cost | python | `block-hash-prefix-cache` | M5 |
| 50 | `hybrid-state-checkpoints` | Prefix Reuse for Recurrent-State Models | hard | numeric, cost | t/j/m | `prefix-cache-reuse` | M5 |
| 60 | `mla-latent-kv-cache` | Caching Latents with Multi-Head Latent Attention | hard | numeric, property | t/j/m | `kv-cache-append`, `rope` | M5 |
| 70 | `speculative-tool-calls` | Speculative Tool Calls | hard | simclock | python | `tool-call-loop`, `tool-call-stream-parser` | M5 |
| 80 | `expert-parallel-dispatch` | Expert Parallelism for MoE | hard | numeric, cost | t/j/m | `moe-sparse-dispatch`, `tensor-parallel-linear` | M5 |

**`reasoning-budget-processor`** — Implement `budget_step(state, logits, cfg) -> (state, logits)`: count tokens generated inside the thinking span (after the start-of-thinking token), and when the budget is reached force the end-of-thinking token sequence one token per step, after which normal sampling and any structured-output constraint resume (vLLM `thinking_token_budget`; structured outputs disabled during reasoning unless `enable_in_reasoning`). Cost: on a `TinyLM` run, thinking tokens never exceed the budget. Mutants: budget counted over all tokens; multi-token end marker forced only at its first token; constraint applied inside the thinking span.

**`cross-layer-kv-sharing`** — Implement `prefill_shared(lm, prompt, share_from)` and `decode_step(...)` for a model whose layers at and above `share_from` reuse the keys and values of layer `share_from - 1`: prefill runs only the lower layers for all prompt tokens except the last, which runs the full stack (YOCO, Sun et al., 2024; the KV-shared layers of Gemma 4 E2B). Cost: `layer_tokens` for prefill equals `share_from * P + (L - share_from)`; logits equal the reference. Mutants: full stack on every prompt token; upper layers keep their own caches; last token skipped. [kit: `LayeredLM`]

**`diffusion-unmask-step`** — Implement `unmask_step(tokens, logits, mask_id, n_unmask, temperature, uniforms)`: for every masked position sample a candidate from the supplied uniforms, compute its confidence, and commit the `n_unmask` most confident positions (ties to the lowest index), leaving the rest masked; plus `schedule(n_masked, steps)` (confidence-based parallel decoding as in MaskGIT, Chang et al., 2022, used by masked-diffusion language models such as LLaDA, Nie et al., 2025). Statistical on the committed token distribution at a single masked position; numeric on worked cases. Mutants: commits already unmasked positions; confidence of the argmax rather than of the sampled token; schedule that leaves masks after the last step.

**`kv-tiered-offload`** — Implement `TieredKV(gpu_blocks, host_blocks, costs)`: on GPU pressure demote least-recently-used unreferenced prefix blocks to the host tier, and on a lookup decide per request between loading blocks back (cost per block) and recomputing (cost per token), choosing the cheaper and never loading a chain with a missing ancestor (LMCache, MLSys 2026; vLLM KV offloading backends). Simclock and cost: total time on the fixture workload within the oracle's bound. Mutants: evict instead of demote; always load even when recompute is cheaper; loads a child block whose parent was dropped.

**`hybrid-state-checkpoints`** — Implement `generate_with_checkpoints(lm, store, prompt, interval)` for a model whose per-layer state is a fixed-size recurrent tensor that cannot be truncated: save state checkpoints every `interval` tokens during prefill, and for a new prompt resume from the deepest checkpoint at or before the common prefix (the hybrid KV manager for Qwen3-Next's Gated DeltaNet layers in vLLM; SGLang's `MambaRadixCache`). Cost: `tokens_processed <= P - floor(lcp / interval) * interval + g - 1`. Mutants: resumes from the final state of a longer cached prompt (wrong logits); checkpoint at the nearest boundary after the prefix; no checkpoints stored during generation. [kit: `RecurrentLM`]

**`mla-latent-kv-cache`** — Implement `mla_step(x_new, params, cache)` for Multi-Head Latent Attention: cache only the compressed latent `c = x W_dkv` and the shared RoPE key part, and compute attention by absorbing the key up-projection into the query and the value up-projection into the output projection, so full keys and values are never materialized per cached token (DeepSeek-V2, 2024, section 2.1). Property: equal to the explicit formulation; cached floats per token equal `d_c + d_rope`. Mutants: caches up-projected keys; RoPE applied to the content part; absorbed matrices multiplied in the wrong order.

**`speculative-tool-calls`** — Implement `async run(model_stream, tools, predictor)`: while the model is still streaming a tool call's arguments, launch the predicted call for side-effect-free tools; when the final arguments arrive, use the speculative result if they match exactly, otherwise cancel it and run the real call; never speculate on tools marked as having side effects (PASTE, arXiv 2603.18897, 2026). Simclock: end-to-end time beats the sequential oracle by the stated margin and no canceled result is ever used. Mutants: speculative result used on a near match; side-effecting tool speculated; mispredicted call not canceled. [kit: `ScriptedChatModel`]

**`expert-parallel-dispatch`** — Implement `ep_forward(x, router, placement, devs)`: experts live on different simulated devices; group routed token rows by destination device, send each group once, run the experts there, and return and combine results in the original order with the routing weights (expert parallelism as in DeepEP and vLLM's wide-EP serving). Cost: one send and one receive per device pair with tokens, `comm_bytes` equal to the routed rows; numeric against the dense oracle. Mutants: one message per token; every token broadcast to every device; combine order wrong when a token's two experts are on the same device. [kit: `SimDevices`, `CountingExperts`]

### Track 15: `capstones` Capstones (order 150)

Multi-part builds. Each is one problem whose cases are grouped into graded stages; a stage's cases are named `Stage N: ...`, earlier stages are public-heavy, and the last stage is hidden-heavy. Each stage adds entry points; the starter contains all signatures. All hard, all M5. `time_limit_s` 30.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `capstone-continuous-batching-engine` | A Continuous-Batching Engine with a Paged KV Cache | hard | simclock, cost, property | python | `continuous-batching-scheduler`, `paged-kv-allocator`, `chunked-prefill-scheduling`, `preemption-recompute` | M5 |
| 20 | `capstone-quantized-runner` | A Quantized Model Runner | hard | numeric, property, cost | t/j/m | `groupwise-4bit`, `quantized-matmul`, `decoder-forward`, `kv-cache-append` | M5 |
| 30 | `capstone-speculative-engine` | A Speculative Decoding Engine | hard | statistical, numeric, cost | t/j/m | `speculative-sampling-block`, `speculative-decode-loop`, `dynamic-draft-length` | M5 |
| 40 | `capstone-structured-output-server` | A Structured-Output Server | hard | service, numeric | python | `openai-chat-endpoint`, `json-schema-regex`, `dfa-token-index`, `tool-call-stream-parser` | M5 |
| 50 | `capstone-real-model-runtime` | A Complete Runtime for a Real Model | hard | numeric, cost | t/j/m | `real-model-logits`, `real-model-cached-generate`, `tokenizer-json-load`, `sampler-chain` | M5 |
| 60 | `capstone-inference-gateway` | An Inference Gateway | hard | simclock | python | `prefix-aware-router`, `admission-control`, `vtc-fair-scheduler`, `autoscaler-policy` | M5 |

**`capstone-continuous-batching-engine`** — Build `Engine(sim_engine, n_blocks, block_size, max_batch_tokens)`. Stage 1: block allocator with fork and free. Stage 2: continuous-batching loop, FCFS. Stage 3: chunked prefill under the token budget. Stage 4: priorities and preemption by recompute under memory pressure. Stage 5: prefix sharing of full blocks between requests with a common prompt. Judged by simclock traces against the oracle per stage, allocator invariants as properties, and the engine's recompute and block counters. Mutants: one per stage (leaked blocks, idle slot, prefill starving decodes, preemption thrash, shared block freed while referenced). [kit: `SimEngine`]

**`capstone-quantized-runner`** — Build a runner for the kit decoder. Stage 1: `quantize_checkpoint(params, bits, group_size)` to packed codes, scales, and biases. Stage 2: `forward_q` using quantized matmuls only. Stage 3: `generate_q` with a KV cache. Stage 4: a mixed recipe (8-bit embeddings and head, 4-bit elsewhere) meeting a bits-per-weight budget and a logit KL limit against the float model. Judged numerically against the oracle at each stage, by error-bound properties, and by a cost check on stored bytes. Mutants: float weights kept alongside codes; cache bypassed; recipe over budget. [kit: `decoder_params`]

**`capstone-speculative-engine`** — Build `SpeculativeEngine(target, draft)`. Stage 1: greedy speculation with cache rollback, output equal to the target's. Stage 2: sampling with exact rejection, statistical judge on the emitted distribution. Stage 3: adaptive draft length meeting a cost bound on mixed-acceptance fixtures. Stage 4: prompt-lookup drafting as a second proposer with a per-step choice between proposers. Stage 5: batch of two sequences with ragged acceptance. Mutants: one per stage (stale cache, residual unnormalized, fixed `k`, lookup drafts unverified, rows truncated to the batch minimum). [kit: `draft_target_pair`]

**`capstone-structured-output-server`** — Build `make_app(lm, vocab)`. Stage 1: `/v1/chat/completions` streaming and non-streaming. Stage 2: `response_format` with a JSON Schema compiled to a token-level constraint, every response validating against the schema even when the scripted model prefers invalid tokens. Stage 3: `tools` with streamed `tool_calls` deltas and `finish_reason: "tool_calls"`. Stage 4: cancellation on client disconnect and a schema-compilation cache keyed by schema hash (the second request with the same schema performs no compilation, by counter). Service judge throughout. Mutants: constraint dropped on the streaming path; invalid JSON at `max_tokens` truncation not flagged with `finish_reason: "length"`; cache keyed by object identity. [kit: `ScriptedLM`, `ToyVocab`, `CharDFA`]

**`capstone-real-model-runtime`** — Build a runtime for `granite-4.0-350m` from raw files. Stage 1: load weights from safetensors by memory map. Stage 2: tokenizer and chat template, exact prompt ids. Stage 3: forward pass matching fixture logits. Stage 4: cached generation with the sampler chain and supplied uniforms, exact fixture tokens. Stage 5: streaming text with stop sequences. All components are the learner's own in one file; nothing is imported from `transformers`, `tokenizers`, or `safetensors`. Mutants: one per stage. Cases are skipped when the asset is absent. [kit: `assets`]

**`capstone-inference-gateway`** — Build `Gateway(cluster, cfg)` in front of simulated replicas. Stage 1: admission control with per-tenant token-rate limits. Stage 2: least-loaded routing. Stage 3: prefix-cache-aware routing with affinity breaking. Stage 4: fair scheduling across tenants at the gateway queue. Stage 5: autoscaling with startup delay, and draining a replica without failing its requests. Judged by simclock on goodput, per-tenant fairness, cache-hit tokens, and replica-seconds against oracle-derived limits. Mutants: one per stage (unbounded queue, request-count balancing, sticky routing to an overloaded replica, tenant starvation, scale-down that kills in-flight requests). [kit: `SimCluster`]

## 5. Author batches for M2

Each batch is one author's unit of work: 4 or 5 problems that share a track or kit
knowledge. Every M2 problem is in exactly one batch. Kit work listed under "needs" must
land before the batch starts; batches with no entry need nothing new.

| batch | theme | problems | needs |
|---|---|---|---|
| 1 | Tensor mechanics | `gelu-silu-activations`, `split-merge-heads`, `attention-mask-builder`, `last-token-gather`, `sequence-log-likelihood` | |
| 2 | Tokenizer edges | `utf8-boundaries`, `pad-batch-left`, `special-tokens-split`, `streaming-detokenizer`, `chat-template-render` | `ToyVocab` |
| 3 | Attention variants | `swiglu-mlp`, `grouped-query-attention`, `qk-norm-attention`, `sliding-window-attention`, `last-position-head` | |
| 4 | Whole forward pass | `pre-norm-block`, `decoder-forward`, `padded-batch-invariance`, `moe-router-forward` | `ops`, `decoder_params` |
| 5 | Single-step samplers | `temperature-scaling`, `top-k-filtering`, `min-p-filtering`, `categorical-sample`, `frequency-presence-penalty` | |
| 6 | Decoding loops | `sampler-chain`, `stop-sequences`, `beam-search`, `batched-cached-decode` | `ops`, `KVCache.select`, `ToyVocab` |
| 7 | Cache tensors | `kv-cache-append`, `cached-rope-offsets`, `preallocated-kv-cache`, `sliding-window-cache` | `ops` |
| 8 | Cache-aware loops | `chunked-prefill-cache`, `prefix-cache-reuse`, `paged-kv-allocator`, `constrained-greedy-decode` | `KVCache.truncate`, `TokenConstraint`, `ToyVocab` |
| 9 | Token masks | `token-mask-apply`, `choice-trie-constraint`, `dfa-token-mask`, `dfa-token-index`, `json-pushdown-validator` | `CharDFA`, `ToyVocab` |
| 10 | Bits and error | `bf16-decode`, `bit-packing`, `symmetric-per-channel`, `quant-error-metrics` | |
| 11 | Group quantization | `groupwise-4bit`, `quantized-matmul`, `clipping-calibration`, `lora-merge` | |
| 12 | Speculative decoding | `greedy-speculative-verify`, `prompt-lookup-draft`, `speculative-speedup-model`, `speculative-sampling-block`, `speculative-decode-loop` | `draft_target_pair`, `KVCache.truncate` |
| 13 | Sizing on paper | `param-count`, `kv-cache-bytes`, `flops-per-token`, `memory-bandwidth-bound`, `latency-percentiles` | |
| 14 | The request path | `latency-metrics`, `token-bucket-limiter`, `admission-control`, `metrics-endpoint`, `openai-chat-endpoint` | `ScriptedLM`, `SimRequest`, `workloads` |
| 15 | Engine scheduling | `continuous-batching-scheduler`, `priority-scheduler`, `deadline-cancellation`, `chunked-prefill-scheduling`, `vtc-fair-scheduler` | `SimEngine`, `SimRequest`, `workloads` |

Suggested order: kit work first (`ops`, `decoder_params`, `ToyVocab`, `KVCache.truncate`
and `select`, then `CharDFA`, `TokenConstraint`, `draft_target_pair`, `SimEngine`,
`ScriptedLM`); batches 1, 3, 5, 10, 11, and 13 need no kit work and can start
immediately. Within batch 15, write `continuous-batching-scheduler` first; the other
four extend its loop and reuse its simulation helper.

## 6. Inference 75

The curated path (`paths/inference-75.toml`): the 75 problems to do, in order, to cover
what a strong inference engineer must be able to write from memory. A dagger (†) marks
problems that do not exist at the end of M2; the path file lists all 75, and the app
skips the missing ones until they ship (58 exist at the end of M2).

| # | stage | one sentence | problems |
|---|---|---|---|
| 1 | Numerics and tensors | Stable arithmetic and the indexing that every later problem assumes. | `stable-softmax`, `rmsnorm`, `attention-mask-builder`, `last-token-gather`, `sequence-log-likelihood` |
| 2 | Text to tokens and back | Encode, decode, and template text exactly as a production tokenizer does. | `bpe-encode`, `special-tokens-split`, `streaming-detokenizer`, `chat-template-render`, `tokenizer-json-load`† |
| 3 | The forward pass | Build a current decoder-only model piece by piece, then match a real one. | `causal-self-attention`, `rope`, `swiglu-mlp`, `grouped-query-attention`, `decoder-forward`, `padded-batch-invariance`, `moe-router-forward`, `real-model-logits`† |
| 4 | Choosing the next token | The samplers engines ship, composed in the right order, and when to stop. | `temperature-scaling`, `top-k-filtering`, `top-p-filtering`, `min-p-filtering`, `repetition-penalty`, `categorical-sample`, `sampler-chain`, `stop-sequences` |
| 5 | The KV cache | Compute each token once, then share, page, and index the memory. | `kv-cache-append`, `cached-greedy-decode`, `cached-rope-offsets`, `preallocated-kv-cache`, `batched-cached-decode`, `prefix-cache-reuse`, `paged-kv-allocator`, `radix-prefix-cache`† |
| 6 | Structured output | Turn a regex or schema into per-step token masks and decode under them. | `token-mask-apply`, `dfa-token-mask`, `dfa-token-index`, `constrained-greedy-decode`, `json-schema-regex`† |
| 7 | Quantization | Shrink weights, compute on the codes, and measure what was lost. | `affine-int8`, `symmetric-per-channel`, `groupwise-4bit`, `quantized-matmul`, `lora-merge`, `gptq-column-update`† |
| 8 | Speculative decoding | Draft cheaply and verify exactly, first for one token, then in a loop. | `greedy-speculative-verify`, `speculative-sampling-step`, `speculative-sampling-block`, `speculative-decode-loop` |
| 9 | Sizing and measurement | Predict memory and speed from a config, and measure latency correctly. | `param-count`, `kv-cache-bytes`, `memory-bandwidth-bound`, `latency-metrics`, `latency-percentiles`, `goodput-slo`† |
| 10 | Serving | Batch, admit, schedule, cancel, and stream under load. | `dynamic-batcher`, `sse-token-stream`, `token-bucket-limiter`, `admission-control`, `continuous-batching-scheduler`, `chunked-prefill-scheduling`, `deadline-cancellation`, `vtc-fair-scheduler`, `openai-chat-endpoint`, `preemption-recompute`†, `prefix-aware-router`† |
| 11 | On the GPU | Lazy evaluation, compilation, and static caches on MLX and CUDA. | `mlx-lazy-eval-placement`†, `mlx-compile-step`†, `mlx-streaming-generate`†, `cuda-sync-free-decode-loop`†, `cuda-static-kv-cache`† |
| 12 | Around the model | Place a model across devices and drive it from an application loop. | `device-map-solver`†, `tool-call-loop`† |
| 13 | Capstones | Put it together: an engine and a complete runtime for a real model. | `capstone-continuous-batching-engine`†, `capstone-real-model-runtime`† |

## 7. Coverage notes and deliberate omissions

- **No training topics.** `bpe-learn-merges` builds a tokenizer; calibration-based
  quantization (`gptq-column-update`, `awq-scale-search`) uses activations but updates
  nothing by gradient. Learned drafters (EAGLE-3, block-diffusion drafters) are covered
  through their inference mechanics (`mtp-head-draft`, `tree-attention-mask`), not their
  training.
- **Dropped as outdated or removed upstream:** tail-free sampling (removed from
  llama.cpp in 2024), contrastive search and group beam search (moved out of
  `transformers` core in v5), learned absolute and sinusoidal positions, ALiBi, GPT-2
  conventions.
- **Not judgeable deterministically, so left out:** MLX streams and threads, MLX
  distributed (`mx.distributed`), real multi-GPU communication, and real wall-clock
  serving benchmarks. Their logic is taught on simulated devices and the simulated
  clock instead.
- **Near-duplicate boundaries.** `sliding-window-attention` is the mask;
  `sliding-window-cache` is the storage. `preallocated-kv-cache` (functional, all
  frameworks), `mlx-preallocated-kv-cache` (step growth and lazy evaluation), and
  `cuda-static-kv-cache` (compile-stable shapes) teach different constraints on the same
  idea and say so in their statements. `prefix-cache-reuse` is one sequence;
  `block-hash-prefix-cache` and `radix-prefix-cache` are the two engine-wide indexes;
  `prefix-aware-router` is the cluster level.

## 8. Decisions and risks for the owner

1. **M3 models.** Granite 4.0 350M plus Granite 4.2 3B (8-bit MLX build from
   `lmstudio-community`) versus the Qwen3 fallback pair (section 2). Granite is newer,
   smaller, and Apache-2.0, but its scaling multipliers make it less "vanilla", and
   native `transformers` support is inferred, not tested.
2. **Harness changes this plan assumes:** per-case `requires="cuda"` (without it the
   CUDA track cannot validate on a Mac); `frameworks = ["mlx"]` with an MLX oracle;
   `assets` in `problem.toml` with skip-when-missing; optional stage grouping.
3. **CUDA kernels.** The two Triton problems rely on a numpy shim of a Triton subset
   (`tritonshim`). It is the largest piece of kit work in the plan; the alternative is
   to cut both and end the track at `cuda-graph-replay` (12 problems).
4. **Catalog size.** The plan has 213 problems against the target of about 210; the
   three extra are M3 real-model problems.

## Sources

Checked 2026-11-14. Items marked (secondary) were confirmed only through a secondary
source; classic papers are cited from the literature and were not re-read on that date.

Models
- Granite: https://huggingface.co/ibm-granite/granite-4.0-350m , https://huggingface.co/ibm-granite/granite-4.2-3b , https://huggingface.co/lmstudio-community/granite-4.2-3b-MLX-8bit , https://huggingface.co/mlx-community/granite-4.0-350m-8bit
- Fallback and rejected candidates: https://huggingface.co/Qwen/Qwen3-0.6B , https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 , https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-8bit , https://huggingface.co/Qwen/Qwen3.5-0.8B , https://huggingface.co/google/gemma-4-E2B-it , https://huggingface.co/google/gemma-3-270m , https://huggingface.co/LiquidAI/LFM2.5-350M , https://huggingface.co/HuggingFaceTB/SmolLM3-3B

Sampling and decoding
- llama.cpp server README (default sampler chain `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`): https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md ; tail-free sampling removed: https://github.com/ggml-org/llama.cpp/pull/10071
- vLLM `SamplingParams`: https://docs.vllm.ai/en/latest/api/vllm/sampling_params/
- transformers generation strategies: https://huggingface.co/docs/transformers/main/en/generation_strategies
- mlx-lm samplers: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py
- Papers: Holtzman et al. 2019 (nucleus); Nguyen et al. 2024 (min-p); Tang et al. 2024 (top-n-sigma); Basu et al. 2020 (mirostat). XTC and DRY are llama.cpp community samplers by p-e-w, with no paper.

Structured output
- vLLM structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs/
- XGrammar: https://github.com/mlc-ai/xgrammar (Dong et al. 2024); XGrammar-2: https://arxiv.org/abs/2601.04426
- llguidance: https://github.com/guidance-ai/llguidance ; llama.cpp GBNF: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- Willard and Louf 2023 (Outlines)

Quantization
- llama.cpp quantize README: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- llm-compressor: https://github.com/vllm-project/llm-compressor ; vLLM quantization: https://docs.vllm.ai/en/latest/features/quantization/
- `mx.quantize` (affine, mxfp4, mxfp8, nvfp4 modes): https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html ; mlx-lm learned quantization: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md
- Papers: Frantar et al. 2022 (GPTQ); Lin et al. 2023 (AWQ); Dettmers et al. 2023 (QLoRA, NF4); Ma et al. 2024 (BitNet b1.58); Hu et al. 2021 (LoRA); OCP Microscaling Formats v1.0 (2023)

Speculative decoding
- vLLM speculative decoding (draft model, eagle3, mtp, ngram, suffix, dynamic K): https://docs.vllm.ai/en/latest/features/speculative_decoding/
- Papers: Leviathan et al. 2023; Chen et al. 2023; Miao et al. 2023 (SpecInfer); Sun et al. 2023 (SpecTr); Cai et al. 2024 (Medusa); Li et al. 2024 (EAGLE-2); Elhoushi et al. 2024 (LayerSkip); Oliaro et al. 2024 (SuffixDecoding); Saxena 2023 (prompt lookup decoding)

KV cache
- Kwon et al. 2023 (PagedAttention); Zheng et al. 2024 (SGLang RadixAttention); Xiao et al. 2023 (StreamingLLM); Zhang et al. 2023 (H2O); Li et al. 2024 (SnapKV)
- mlx-lm caches: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py
- llama.cpp host-memory prompt cache: https://github.com/ggml-org/llama.cpp/pull/16391
- Hybrid-model state caching: https://vllm.ai/blog/qwen3-next ; https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/
- LMCache (MLSys 2026): https://mlsys.org/virtual/2026/oral/3646

Serving
- vLLM releases (v0.33.0; queue caps, P/D connectors): https://github.com/vllm-project/vllm/releases ; scheduler config (secondary, older doc version): https://docs.vllm.ai/en/v0.10.2/api/vllm/config/scheduler.html
- TGI maintenance mode: https://huggingface.co/docs/text-generation-inference/index
- NVIDIA Dynamo: https://docs.nvidia.com/dynamo/latest/reference/releases/v1-3-0
- llm-d scheduling and autoscaling: https://llm-d.ai/docs/guide/Installation/inference-scheduling ; https://llm-d.ai/docs/architecture/advanced/autoscaling
- Ray Serve prefix-aware router: https://docs.ray.io/en/latest/serve/llm/prefix-aware-request-router.html
- Metric definitions: https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference ; https://blog.vllm.ai/guidellm/main/guides/metrics
- Papers: Yu et al. 2022 (Orca); Sheng et al. 2024 (VTC); Zhong et al. 2024 (DistServe); Agrawal et al. 2024 (Sarathi-Serve); Sheng et al. 2023 (S-LoRA); Mitzenmacher 2001

MLX and PyTorch
- MLX 0.32.4 docs: https://ml-explore.github.io/mlx/build/html/python/memory_management.html ; https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html ; https://ml-explore.github.io/mlx/build/html/python/fast.html ; https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html ; https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html ; https://ml-explore.github.io/mlx/build/html/python/data_types.html
- mlx-lm generation (`stream_generate`, `async_eval`, `prefill_step_size`, KV quantization options): https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
- PyTorch 2.15 releases: https://github.com/pytorch/pytorch/releases ; static KV cache with `torch.compile`: https://huggingface.co/docs/transformers/main/en/llm_optims
- Not re-verified on this date (verify when authoring M4): `torch.cuda.memory_stats` keys, Dynamo compile counters, `torch.cuda.CUDAGraph` details, the Triton language subset.

Frontier
- DeepSeek-V2 (2024), Multi-Head Latent Attention; Sun et al. 2024 (YOCO); Chang et al. 2022 (MaskGIT); Nie et al. 2025 (LLaDA); Mercury 2 launch (secondary): https://the-decoder.com/inception-launches-mercury-2-the-first-diffusion-based-language-reasoning-model/
- Speculative tool calls: arXiv 2603.18897 (secondary, abstract only)
- Wide expert parallelism (secondary): https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs
