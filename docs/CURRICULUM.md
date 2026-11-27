# Learn AI curriculum

The complete catalog plan: every problem Learn AI will ship, with the milestone in which
each is authored. Sections 1 to 8 are the fifteen tracks of the "Inference and serving"
area (213 problems); section 9 is the five tracks of the "Model architectures" area (24
problems), added on 2026-11-21. The catalog has 237 problems in 20 tracks. Authors write problems from this document, five at
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
| diff | `difficulty` and `level` in `problem.toml`: `easy`, `medium`, or `hard`, then the level within it, 1 to 3. |
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
| 1 | `foundations` | Foundations | 13 | 2 | 6 | 2 | 0 | 3 | 8 | 5 | 0 |
| 2 | `tokenization` | Tokenization | 13 | 1 | 5 | 2 | 0 | 5 | 4 | 6 | 3 |
| 3 | `transformer` | Transformer Forward Pass | 19 | 2 | 9 | 2 | 0 | 6 | 12 | 7 | 0 |
| 4 | `sampling` | Sampling and Decoding | 16 | 2 | 8 | 0 | 0 | 6 | 10 | 4 | 2 |
| 5 | `kv-cache` | KV Cache and Attention Memory | 16 | 1 | 8 | 1 | 0 | 6 | 6 | 8 | 2 |
| 6 | `constrained` | Constrained Generation | 13 | 0 | 6 | 0 | 0 | 7 | 3 | 4 | 6 |
| 7 | `quantization` | Quantization and Compression | 18 | 1 | 7 | 2 | 0 | 8 | 10 | 6 | 2 |
| 8 | `speculative` | Speculative and Accelerated Decoding | 14 | 1 | 5 | 0 | 0 | 8 | 8 | 3 | 3 |
| 9 | `sizing` | Sizing and Benchmarking | 11 | 0 | 6 | 1 | 0 | 4 | 7 | 3 | 1 |
| 10 | `mlx` | MLX GPU Programming | 18 | 0 | 0 | 18 | 0 | 0 | 5 | 8 | 5 |
| 11 | `cuda` | CUDA GPU Programming | 14 | 0 | 0 | 0 | 14 | 0 | 5 | 8 | 1 |
| 12 | `serving` | Serving Systems | 22 | 2 | 9 | 0 | 0 | 11 | 9 | 9 | 4 |
| 13 | `systems` | Distributed and LLM Application Systems | 12 | 0 | 0 | 0 | 0 | 12 | 4 | 7 | 1 |
| 14 | `frontier` | Frontier | 8 | 0 | 0 | 0 | 0 | 8 | 0 | 4 | 4 |
| 15 | `capstones` | Capstones | 6 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 6 |
| | **all** | | **213** | **12** | **69** | **28** | **14** | **90** | **91** | **82** | **40** |

The table covers the fifteen tracks of sections 4 to 7. The five tracks of section 9
(milestone M6) add 24 problems, 11 easy, 11 medium, and 2 hard: `recurrent-mixers` 6,
`sparse-attention` 5, `block-components` 7, `drafts-and-diffusion` 3, `real-hybrid-model`
3. With them the catalog has 237 problems: 102 easy, 93 medium, 42 hard.

Difficulty split: 43% easy, 39% medium, 18% hard. Each difficulty has three levels (`level` in `problem.toml`); how a problem is placed is in `docs/PROBLEM_AUTHORING.md` section 2. At the end of M2 the catalog has 81 problems. M3 is the 18 MLX problems plus 10 real-model problems in other tracks (two more real-model problems are inside the MLX track).

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

**Verified in milestone 3.** Both models load natively in `transformers` and `mlx-lm`,
and a from-scratch forward pass matches the reference logits (largest error 8.4e-5).
The pinned revisions, the exact place of every multiplier, the 8-bit checkpoint layout,
and the tokenizer details are recorded in `docs/PROBLEM_AUTHORING.md` section 15, which
supersedes this table where they differ. Known differences: the two tokenizers differ
in pre-tokenizer and special tokens; the small model's chat format is
`<|start_of_role|>user<|end_of_role|>...<|end_of_text|>`, not ChatML; `mlx-lm` runs the
model in bfloat16, so compare against it with a tolerance of about 0.4 in logits or
load float32. Granite was chosen; the fallback below was tested and not needed.

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
on demand into `.learnai/models/` (`learnai assets fetch <name>`), and a problem whose assets are
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
| `ops` | `learnai.kit.ops`: framework-native reference building blocks, one signature in all frameworks: `softmax`, `rms_norm`, `rope`, `attention(q, k, v, mask)`, `swiglu`, `top_k_filter`, `top_p_filter`, `min_p_filter`. Lets a later problem use an earlier concept without re-deriving it. | `pre-norm-block`, `sampler-chain` |
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
| `assets` | `assets.path("granite-4.0-350m", "model.safetensors") -> Path` or raises `AssetMissing`; registry of repo id, revision hash, file list, sizes, SHA-256; CLI `learnai assets fetch <name>`. As built: `[[assets]]` tables in `problem.toml`, `ctx.asset(name, file)` in tests, and cases that need a missing asset are skipped (see PROBLEM_AUTHORING section 15). | every real-model problem |
| `fixtures` | `ctx.fixture("logits.npz")` loads a file from the problem's `fixtures/` directory. | every real-model problem |
| MLX-only oracle | `learnai validate` accepts `frameworks = ["mlx"]` with `reference/mlx.py` as the oracle (today a tensor problem must list `torch`). | track `mlx` |
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
| `tritonshim` | `from learnai.kit import tritonshim as tl`: a numpy implementation of the subset of `triton.language` the track uses (`program_id`, `arange`, `load`, `store` with masks, `where`, `sum`, `max`, `sqrt`, `exp`), plus `launch(kernel, grid, **args)`. On a CUDA machine with Triton installed the same kernel source is compiled by real Triton. | `cuda-triton-rmsnorm` |

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

### Track 1: `foundations` Foundations (order 10)

The numerics and tensor mechanics every later track leans on.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `stable-softmax` | Numerically Stable Softmax | easy 2 | numeric | t/j/m | - | M1 |
| 20 | `rmsnorm` | RMSNorm | easy 3 | numeric, property | t/j/m | - | M1 |
| 30 | `gelu-silu-activations` | GELU and SiLU | easy 1 | numeric | t/j/m | - | M2 |
| 40 | `split-merge-heads` | Splitting and Merging Attention Heads | easy 2 | numeric, property | t/j/m | - | M2 |
| 50 | `attention-mask-builder` | Causal and Padding Masks | easy 3 | numeric | t/j/m | - | M2 |
| 60 | `last-token-gather` | Gathering the Last Real Token | easy 2 | numeric | t/j/m | - | M2 |
| 70 | `sequence-log-likelihood` | Sequence Log-Likelihood and Perplexity | easy 3 | numeric | t/j/m | `stable-softmax` | M2 |
| 80 | `bf16-decode` | Decoding bfloat16 by Hand | medium 1 | numeric, property | t/j/m | - | M2 |
| 90 | `layernorm` | LayerNorm | easy 1 | numeric, property | t/j/m | `rmsnorm` | M5 |
| 100 | `kl-divergence-from-logits` | KL Divergence Between Logit Sets | medium 1 | numeric, property | t/j/m | `stable-softmax` | M5 |
| 110 | `half-precision-rmsnorm` | RMSNorm That Survives float16 | medium 3 | numeric | t/j/m | `rmsnorm` | M5 |
| 120 | `safetensors-parse` | Parsing safetensors by Hand | medium 1 | numeric | python | `bf16-decode` | M3 |
| 130 | `real-model-perplexity` | Perplexity of a Real Model | medium 1 | numeric | t/j/m | `sequence-log-likelihood` | M3 |

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
| 10 | `bpe-encode` | Byte-Level BPE Encoding | medium 1 | numeric, property | python | - | M1 |
| 20 | `utf8-boundaries` | UTF-8 Character Boundaries | medium 2 | numeric, property | python | - | M2 |
| 30 | `pad-batch-left` | Padding a Batch for Generation | easy 2 | numeric | python | - | M2 |
| 40 | `special-tokens-split` | Special Tokens Without Injection | easy 3 | numeric | python | `bpe-encode` | M2 |
| 50 | `streaming-detokenizer` | Streaming Detokenization | medium 2 | numeric, property, cost | python | `utf8-boundaries` | M2 |
| 60 | `chat-template-render` | Rendering a Chat Template | medium 1 | numeric | python | `special-tokens-split` | M2 |
| 70 | `wordpiece-encode` | WordPiece Encoding | easy 2 | numeric | python | - | M5 |
| 80 | `context-window-trim` | Fitting a Conversation into the Context Window | easy 3 | numeric, property | python | `chat-template-render` | M5 |
| 90 | `bpe-learn-merges` | Learning a BPE Merge Table | medium 1 | numeric | python | `bpe-encode` | M5 |
| 100 | `bpe-encode-fast` | BPE Encoding in O(n log n) | hard 2 | numeric, speed | python | `bpe-encode` | M5 |
| 110 | `unigram-viterbi` | Unigram Tokenization with Viterbi | hard 1 | numeric, property | python | - | M5 |
| 120 | `tokenizer-json-load` | Running a Real Tokenizer's Merges | hard 1 | numeric | python | `bpe-encode`, `special-tokens-split` | M3 |
| 130 | `real-model-chat-turn` | One Chat Turn with a Real Model | medium 2 | numeric | python | `chat-template-render`, `streaming-detokenizer` | M3 |

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
| 10 | `causal-self-attention` | Causal Self-Attention | easy 3 | numeric, property | t/j/m | - | M1 |
| 20 | `rope` | Rotary Position Embeddings (RoPE) | easy 3 | numeric, property | t/j/m | - | M1 |
| 30 | `swiglu-mlp` | SwiGLU Feed-Forward | easy 1 | numeric | t/j/m | `gelu-silu-activations` | M2 |
| 40 | `grouped-query-attention` | Grouped-Query Attention | easy 3 | numeric, property | t/j/m | `causal-self-attention` | M2 |
| 50 | `pre-norm-block` | A Pre-Norm Decoder Block | easy 2 | numeric | t/j/m | `causal-self-attention`, `rmsnorm` | M2 |
| 60 | `decoder-forward` | Full Decoder Forward Pass | medium 1 | numeric, property | t/j/m | `pre-norm-block`, `rope` | M2 |
| 70 | `last-position-head` | Unembedding Only What You Need | easy 1 | numeric, speed | t/j/m | - | M2 |
| 80 | `sliding-window-attention` | Sliding-Window Attention | easy 3 | numeric, property | t/j/m | `causal-self-attention` | M2 |
| 90 | `qk-norm-attention` | QK-Norm Attention | easy 3 | numeric | t/j/m | `causal-self-attention`, `rmsnorm` | M2 |
| 100 | `padded-batch-invariance` | Batching Without Changing the Answer | medium 1 | numeric, property | t/j/m | `decoder-forward`, `attention-mask-builder` | M2 |
| 110 | `moe-router-forward` | Mixture-of-Experts Layer | easy 3 | numeric | t/j/m | `swiglu-mlp` | M2 |
| 120 | `embedding-scale-softcap` | Embedding Scaling and Logit Soft-Capping | easy 1 | numeric | t/j/m | - | M5 |
| 130 | `fused-projection-split` | Splitting Fused Projections | easy 2 | numeric, property | t/j/m | `swiglu-mlp` | M5 |
| 140 | `rope-layout-conversion` | Interleaved versus Half-Split RoPE | medium 1 | numeric, property | t/j/m | `rope` | M5 |
| 150 | `attention-sinks-softmax` | Attention with Learned Sinks | easy 3 | numeric, property | t/j/m | `causal-self-attention` | M5 |
| 160 | `moe-sparse-dispatch` | Sparse Expert Dispatch | medium 2 | numeric, cost | t/j/m | `moe-router-forward` | M5 |
| 170 | `online-softmax-attention` | Blockwise Attention with Online Softmax | medium 2 | numeric, property | t/j/m | `causal-self-attention`, `stable-softmax` | M5 |
| 180 | `real-model-block` | One Block of a Real Model | medium 3 | numeric | t/j/m | `grouped-query-attention`, `swiglu-mlp`, `rope` | M3 |
| 190 | `real-model-logits` | Matching a Real Model's Logits | medium 3 | numeric | t/j/m | `real-model-block`, `decoder-forward` | M3 |

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
| 10 | `top-p-filtering` | Nucleus (Top-p) Filtering | easy 3 | numeric, property | t/j/m | - | M1 |
| 20 | `repetition-penalty` | Vectorized Repetition Penalty | easy 3 | numeric, speed | t/j/m | - | M1 |
| 30 | `temperature-scaling` | Temperature | easy 2 | numeric, property | t/j/m | `stable-softmax` | M2 |
| 40 | `top-k-filtering` | Top-k Filtering | easy 3 | numeric, property | t/j/m | - | M2 |
| 50 | `categorical-sample` | Inverse-CDF Sampling | easy 2 | numeric, statistical | t/j/m | `stable-softmax` | M2 |
| 60 | `min-p-filtering` | Min-p Filtering | easy 1 | numeric, property | t/j/m | - | M2 |
| 70 | `frequency-presence-penalty` | Frequency and Presence Penalties | easy 3 | numeric | t/j/m | `repetition-penalty` | M2 |
| 80 | `sampler-chain` | A Sampler Chain in the Right Order | medium 1 | numeric, statistical | t/j/m | `top-p-filtering`, `top-k-filtering`, `temperature-scaling` | M2 |
| 90 | `stop-sequences` | Stop Sequences Across Token Boundaries | medium 1 | numeric, property | python | `streaming-detokenizer` | M2 |
| 100 | `beam-search` | Beam Search with a KV Cache | hard 1 | numeric, cost | t/j/m | `cached-greedy-decode` | M2 |
| 110 | `top-n-sigma` | Top-n-sigma Filtering | easy 3 | numeric | t/j/m | - | M5 |
| 120 | `gumbel-max-sample` | Gumbel-Max Sampling | easy 2 | numeric, statistical | t/j/m | `categorical-sample` | M5 |
| 130 | `logit-bias-bad-words` | Logit Bias and Banned Sequences | easy 3 | numeric | t/j/m | - | M5 |
| 140 | `xtc-sampler` | XTC: Exclude Top Choices | medium 1 | numeric, statistical | t/j/m | `min-p-filtering` | M5 |
| 150 | `mirostat-v2` | Mirostat v2 | medium 1 | numeric, property | t/j/m | `categorical-sample` | M5 |
| 160 | `dry-penalty` | DRY Repetition Penalty | hard 1 | numeric, speed | python | `repetition-penalty` | M5 |

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
| 10 | `cached-greedy-decode` | Greedy Decoding with a KV Cache | easy 3 | cost, numeric | t/j/m | - | M1 |
| 20 | `kv-cache-append` | Attention with an Appended Cache | easy 3 | numeric, property | t/j/m | `causal-self-attention` | M2 |
| 30 | `chunked-prefill-cache` | Prefill in Chunks | easy 1 | numeric, cost | t/j/m | `cached-greedy-decode` | M2 |
| 40 | `cached-rope-offsets` | RoPE Positions Under a Cache | easy 2 | numeric, property | t/j/m | `rope`, `kv-cache-append` | M2 |
| 50 | `preallocated-kv-cache` | A Preallocated KV Cache | easy 3 | numeric, property | t/j/m | `kv-cache-append` | M2 |
| 60 | `batched-cached-decode` | Batched Decoding with Early Finish | medium 1 | numeric, cost | t/j/m | `cached-greedy-decode` | M2 |
| 70 | `prefix-cache-reuse` | Reusing a Cached Prefix | medium 1 | numeric, cost | t/j/m | `cached-greedy-decode` | M2 |
| 80 | `sliding-window-cache` | A Rotating Cache That Keeps Its Sinks | medium 2 | numeric, property | t/j/m | `preallocated-kv-cache` | M2 |
| 90 | `paged-kv-allocator` | A Paged KV Block Allocator | easy 3 | numeric, property | python | - | M2 |
| 100 | `paged-kv-copy-on-write` | Forking Sequences with Copy-on-Write | medium 3 | numeric, property | python | `paged-kv-allocator` | M5 |
| 110 | `block-hash-prefix-cache` | Automatic Prefix Caching by Block Hash | medium 3 | numeric, property | python | `paged-kv-allocator`, `prefix-cache-reuse` | M5 |
| 120 | `radix-prefix-cache` | A Radix-Tree Prefix Cache | hard 2 | numeric, property | python | `prefix-cache-reuse` | M5 |
| 130 | `kv-cache-quantize` | An 8-bit KV Cache | medium 2 | numeric, property | t/j/m | `affine-int8`, `kv-cache-append` | M5 |
| 140 | `kv-eviction-heavy-hitters` | Heavy-Hitter KV Eviction | medium 2 | numeric, property | t/j/m | `kv-cache-append` | M5 |
| 150 | `snapkv-compress` | Compressing a Prompt Cache by Attention Votes | medium 1 | numeric | t/j/m | `kv-eviction-heavy-hitters` | M5 |
| 160 | `real-model-cached-generate` | Cached Generation on a Real Model | hard 1 | numeric, speed | t/j/m | `kv-cache-append`, `cached-rope-offsets`, `real-model-logits` | M3 |

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
| 10 | `token-mask-apply` | Applying a Token Mask | easy 1 | numeric, property | t/j/m | `stable-softmax` | M2 |
| 20 | `choice-trie-constraint` | Constraining Output to a Set of Choices | easy 3 | numeric | python | - | M2 |
| 30 | `dfa-token-mask` | Token Masks from a Character DFA | easy 2 | numeric, property | python | - | M2 |
| 40 | `dfa-token-index` | Precomputing the Token Index | medium 3 | numeric, cost | python | `dfa-token-mask` | M2 |
| 50 | `json-pushdown-validator` | An Incremental JSON Validator | hard 2 | numeric, property | python | - | M2 |
| 60 | `constrained-greedy-decode` | Constrained Decoding Loop | medium 1 | numeric, cost | t/j/m | `cached-greedy-decode`, `token-mask-apply` | M2 |
| 70 | `regex-to-dfa` | Compiling a Regex to a DFA | hard 3 | numeric, property | python | `dfa-token-mask` | M5 |
| 80 | `json-schema-regex` | JSON Schema to Regex | hard 1 | numeric, property | python | `regex-to-dfa` | M5 |
| 90 | `jump-forward-decode` | Jump-Forward Decoding | medium 2 | numeric, cost | t/j/m | `constrained-greedy-decode` | M5 |
| 100 | `token-healing` | Token Healing | medium 2 | numeric | python | `bpe-encode` | M5 |
| 110 | `tool-call-stream-parser` | Parsing Tool Calls from a Stream | hard 2 | numeric, property | python | `json-pushdown-validator` | M5 |
| 120 | `grammar-token-mask` | Token Masks for a Context-Free Grammar | hard 3 | numeric, property | python | `dfa-token-mask`, `json-pushdown-validator` | M5 |
| 130 | `structural-tag-dispatch` | Triggered Constraints with Structural Tags | hard 2 | numeric, property | python | `constrained-greedy-decode` | M5 |

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
| 10 | `affine-int8` | Affine int8 Quantization | easy 3 | numeric, property | t/j/m | - | M1 |
| 20 | `symmetric-per-channel` | Symmetric Per-Channel Weights | easy 2 | numeric, property | t/j/m | `affine-int8` | M2 |
| 30 | `quant-error-metrics` | Measuring Quantization Error | easy 1 | numeric | t/j/m | - | M2 |
| 40 | `bit-packing` | Packing 4-bit Codes | easy 3 | numeric, property | t/j/m | - | M2 |
| 50 | `groupwise-4bit` | Group-Wise 4-bit Quantization | easy 3 | numeric, property | t/j/m | `affine-int8` | M2 |
| 60 | `quantized-matmul` | Multiplying Without Dequantizing | easy 3 | numeric, property | t/j/m | `groupwise-4bit` | M2 |
| 70 | `clipping-calibration` | Calibrating a Clipping Range | easy 3 | numeric, property | t/j/m | `affine-int8`, `quant-error-metrics` | M2 |
| 80 | `lora-merge` | Merging a LoRA Adapter | easy 1 | numeric, property | t/j/m | - | M2 |
| 90 | `ternary-bitnet` | Ternary Weights | easy 3 | numeric, property | t/j/m | - | M5 |
| 100 | `int8-linear-w8a8` | An Integer-Only Linear Layer | medium 1 | numeric | t/j/m | `symmetric-per-channel` | M5 |
| 110 | `nf4-codebook` | NormalFloat 4-bit | easy 3 | numeric | t/j/m | `groupwise-4bit` | M5 |
| 120 | `mxfp4-block-encode` | Microscaling FP4 Blocks | hard 1 | numeric, property | t/j/m | `groupwise-4bit` | M5 |
| 130 | `bit-allocation` | Mixed-Precision Bit Allocation | hard 1 | numeric, property | python | `quant-error-metrics` | M5 |
| 140 | `awq-scale-search` | Activation-Aware Scaling | medium 1 | numeric, property | t/j/m | `groupwise-4bit`, `quant-error-metrics` | M5 |
| 150 | `k-quant-superblock` | GGUF K-Quant Superblocks | medium 3 | numeric, property | t/j/m | `groupwise-4bit`, `bit-packing` | M5 |
| 160 | `gptq-column-update` | GPTQ Error Propagation | medium 1 | numeric, property | t/j/m | `symmetric-per-channel` | M5 |
| 170 | `quantize-real-weights` | Quantizing Real Weights | medium 3 | numeric | t/j/m | `groupwise-4bit`, `quant-error-metrics` | M3 |
| 180 | `mlx-quantized-checkpoint-load` | Reading an 8-bit Checkpoint | medium 2 | numeric | python | `bit-packing`, `safetensors-parse` | M3 |

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
| 10 | `speculative-sampling-step` | Speculative Sampling: One Verification Step | easy 2 | statistical, numeric | t/j/m | - | M1 |
| 20 | `greedy-speculative-verify` | Verifying a Greedy Draft | easy 2 | numeric | t/j/m | - | M2 |
| 30 | `prompt-lookup-draft` | Drafting by Prompt Lookup | easy 3 | numeric | python | - | M2 |
| 40 | `speculative-speedup-model` | Expected Speedup of Speculation | easy 1 | numeric | python | - | M2 |
| 50 | `speculative-sampling-block` | Verifying a Block of Sampled Drafts | easy 3 | statistical, numeric | t/j/m | `speculative-sampling-step` | M2 |
| 60 | `speculative-decode-loop` | A Speculative Decoding Loop | hard 1 | numeric, cost | t/j/m | `greedy-speculative-verify`, `cached-greedy-decode` | M2 |
| 70 | `dynamic-draft-length` | Adapting the Draft Length | medium 1 | numeric, cost | python | `speculative-decode-loop` | M5 |
| 80 | `typical-acceptance` | Lossy Verification by Typical Acceptance | easy 3 | numeric | t/j/m | `greedy-speculative-verify` | M5 |
| 90 | `batched-speculative-accept` | Ragged Acceptance in a Batch | medium 2 | numeric, property | t/j/m | `greedy-speculative-verify` | M5 |
| 100 | `mtp-head-draft` | Drafting with Multi-Token Prediction Heads | easy 3 | numeric | t/j/m | `greedy-speculative-verify` | M5 |
| 110 | `tree-attention-mask` | Verifying a Token Tree in One Pass | medium 2 | numeric, property | t/j/m | `attention-mask-builder`, `greedy-speculative-verify` | M5 |
| 120 | `multi-draft-rejection` | Exact Sampling from Several Drafts | easy 3 | statistical, numeric | t/j/m | `speculative-sampling-step` | M5 |
| 130 | `early-exit-self-speculation` | Self-Speculation by Early Exit | hard 1 | numeric, cost | t/j/m | `speculative-decode-loop` | M5 |
| 140 | `suffix-automaton-draft` | Drafting from a Suffix Automaton | hard 3 | numeric, speed | python | `prompt-lookup-draft` | M5 |

**`greedy-speculative-verify`** — Implement `verify_greedy(draft_tokens, target_logits) -> (n_accepted, next_token)` where `target_logits[i]` scores the position of `draft_tokens[i]` and there is one extra row: accept the longest prefix on which the target argmax equals the draft, and return the target's token at the first mismatch (or the bonus token after a full match). Numeric. Mutants: accepts matches after the first mismatch; no bonus token; off-by-one alignment of logits and drafts.

**`prompt-lookup-draft`** — Implement `lookup_draft(context, max_ngram, min_ngram, k)`: for `n` from `max_ngram` down to `min_ngram`, find the most recent earlier occurrence of the last `n` tokens and return up to `k` tokens that followed it; empty if none (prompt lookup decoding, Saxena, 2023; the `ngram` method in vLLM). Numeric. Mutants: first occurrence; matches the suffix with itself; shorter n-gram preferred.

**`speculative-speedup-model`** — Implement `expected_tokens(alpha, k)` `= (1 - alpha**(k+1)) / (1 - alpha)`, `speedup(alpha, k, c)` with draft-to-target cost ratio `c` (Leviathan et al., 2023, Theorem 3.8), and `best_k(alpha, c, k_max)`. Numeric, including `alpha = 1`. Mutants: geometric series without the bonus token; cost `k * c` without the target call; division by zero at `alpha = 1`.

**`speculative-sampling-block`** — Implement `verify_block(p, q, draft_tokens, u_accept, u_resample) -> (n_accepted, next_token)` with `p` of shape `(K + 1, V)`, `q` of shape `(K, V)`: accept drafts in order with probability `min(1, p/q)`, at the first rejection sample from the normalized residual `max(p - q, 0)`, and after `K` acceptances sample from `p[K]` (Leviathan et al., 2023, Algorithm 1). Statistical judge on the first emitted token and on the token following an accepted prefix. Mutants: continues after a rejection; bonus token sampled from the residual; one uniform reused for acceptance and resampling.

**`speculative-decode-loop`** — Implement `speculative_greedy(target, draft, prompt, k, max_new_tokens, eos_id)`: the draft proposes `k` tokens through its cache, the target scores all of them in one call, accepted tokens are kept, and both caches are truncated to the accepted length before the next round. Output must equal the target's own greedy decode. Cost: one target call per round, target `tokens_processed <= P + rounds * (k + 1)`, draft `tokens_processed <= P + rounds * (k + 1)`; rebuilding either cache from scratch exceeds both, and on the high-agreement fixture the target must make fewer than `g / 2` calls. Mutants: cache not rolled back after rejection; draft continues from a rejected token; EOS inside an accepted run ignored; `max_new_tokens` overshoot. [kit: `draft_target_pair`, `KVCache.truncate`]

**`dynamic-draft-length`** — Implement `next_k(k, n_accepted, n_drafted, cfg)` for the two specified policies: the additive heuristic of Hugging Face assisted generation (all accepted: `k + 2`; otherwise `max(1, k - 1)`), and confidence stopping (`should_stop_drafting(draft_prob, threshold)`); then `simulate(policy, acceptance_trace, costs)` returning total cost. Cost judge: on fixture traces the adaptive policy must beat every fixed `k` by the stated margin. Mutants: grows on any acceptance; never shrinks below the initial `k`; confidence compared on logits.

**`typical-acceptance`** — Implement `typical_accept(p, draft_tokens, epsilon, delta)`: accept draft token `x` while `p[x] >= min(epsilon, delta * exp(-H(p)))`, stopping at the first failure, with the first token always accepted greedily (Medusa, Cai et al., 2024; the entropy-gated family that includes vLLM's lossy verification). Numeric. Mutants: entropy in bits; `max` in place of `min`; continues after a failure. The statement says plainly that this is not distribution-preserving.

**`batched-speculative-accept`** — Implement `advance_batch(tokens, lengths, drafts, n_accepted, next_tokens)`: rows accept different numbers of draft tokens; return the updated padded token matrix, new lengths, and per-row cache lengths to truncate to. Property: each row equals its own single-sequence result. Mutants: all rows truncated to the batch minimum (correct but wasteful, rejected by a count check); padding written inside accepted spans; lengths off by the bonus token.

**`mtp-head-draft`** — Implement `mtp_draft(hidden, heads, w_out, k)`: head `i` maps the last hidden state to the hidden state predicting token `t + 1 + i` (a residual MLP as specified), tokens are the argmax through the shared output head; then verify with kit `ops` (Medusa-style heads; multi-token prediction modules as shipped with DeepSeek-V3 and served by vLLM's `mtp` method). Numeric. Mutants: every head reads the previous head's output when the spec is parallel heads; head index off by one; separate unembedding assumed. As built (corrected 2026-11-21): the heads are Medusa's, parallel and each with its **own** unembedding, and the base token alone goes through the model's output head, so the unembedding mutant is the reverse of the one named here (every head projected through the shared head). The parenthesis above is wrong about DeepSeek-V3: its report describes chained modules that share the embedding and the output head, which is the design of `mtp-chain-draft`; the problem's editorial states the contrast correctly.

**`tree-attention-mask`** — Implement `tree_inputs(parents, prefix_len) -> (mask, position_ids)` for a draft token tree in topological order (each node attends to the prefix and to its ancestors only; position is prefix length plus depth), and `verify_tree(parents, tokens, target_argmax)` returning the longest accepted root path (SpecInfer, Miao et al., 2023; EAGLE-2, Li et al., 2024). Property: each node's mask row equals the mask of its path as a linear sequence. Mutants: siblings visible; positions by flat index; path chosen by depth-first order rather than acceptance.

**`multi-draft-rejection`** — Implement `multi_draft_step(p, q, candidates, u_accept, u_resample)` for `n` i.i.d. candidates drawn from `q`: recursive rejection sampling, accepting candidate `i` with `min(1, p_i / q)` and replacing `p` by the normalized residual after each rejection, finally sampling from the last residual (SpecInfer, Miao et al., 2023; analyzed in SpecTr, Sun et al., 2023). Statistical judge: output follows `p` for any `n`. Mutants: residual not updated between candidates; first accepted of independent tests against the original `p`; residual left unnormalized.

**`early-exit-self-speculation`** — Implement `self_speculative_greedy(lm, prompt, exit_layer, k, max_new_tokens)`: draft tokens with the first `exit_layer` layers through the shared head, then verify by running only the remaining layers on the draft positions, reusing the early layers' hidden states and cache (LayerSkip, Elhoushi et al., 2024). Output equals full-model greedy. Cost: `layer_tokens` under a bound that recomputing the early layers during verification exceeds. Mutants: early layers recomputed; late-layer cache not rolled back; exit hidden state unnormalized before the head. [kit: `LayeredLM`]

**`suffix-automaton-draft`** — Implement `SuffixIndex` with `extend(token)` (online construction of a suffix automaton over all previously generated outputs and the current prompt) and `draft(context, max_len, min_match)` returning the continuation after the longest suffix match, chosen by occurrence count among candidates (SuffixDecoding, Oliaro et al., 2024; the `suffix` method in vLLM). Speed: amortized O(1) `extend` and O(match) `draft`, within 5x of the reference on 100,000 tokens. Mutants: rescans history per draft; shortest match; counts not propagated along suffix links.

### Track 9: `sizing` Sizing and Benchmarking (order 90)

Predict memory and speed on paper, then measure the right thing.

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `param-count` | Counting Parameters | easy 3 | numeric | python | - | M2 |
| 20 | `kv-cache-bytes` | KV Cache Bytes per Token | easy 2 | numeric | python | - | M2 |
| 30 | `latency-metrics` | TTFT, ITL, and TPOT | easy 2 | numeric | python | - | M2 |
| 40 | `latency-percentiles` | Percentiles Done Right | easy 3 | numeric, property | python | - | M2 |
| 50 | `flops-per-token` | FLOPs for Prefill and Decode | easy 3 | numeric | python | `param-count` | M2 |
| 60 | `memory-bandwidth-bound` | The Memory-Bandwidth Bound on Tokens per Second | medium 1 | numeric | python | `param-count`, `kv-cache-bytes` | M2 |
| 70 | `memory-budget-fit` | How Many Sequences Fit | easy 3 | numeric | python | `kv-cache-bytes` | M5 |
| 80 | `goodput-slo` | Goodput Under an SLO | easy 3 | numeric | python | `latency-metrics`, `latency-percentiles` | M5 |
| 90 | `stream-latency-probe` | Measuring a Token Stream | hard 1 | simclock | python | `latency-metrics` | M5 |
| 100 | `capacity-planner` | Capacity Planning from a Cost Model | medium 3 | numeric, property | python | `goodput-slo` | M5 |
| 110 | `real-model-sizing` | Sizing a Real Checkpoint | medium 3 | numeric | python | `param-count`, `kv-cache-bytes`, `safetensors-parse` | M3 |

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
| 10 | `mlx-lazy-eval-placement` | Where to Evaluate | easy 3 | numeric, cost | mlx | `cached-greedy-decode` | M3 |
| 20 | `mlx-mask-free-indexing` | Filtering Without Boolean Indexing | medium 1 | numeric | mlx | `top-k-filtering` | M3 |
| 30 | `mlx-dtype-discipline` | Staying in Half Precision | easy 2 | numeric, cost | mlx | - | M3 |
| 40 | `mlx-memory-accounting` | Measuring Unified Memory | medium 1 | numeric, cost | mlx | - | M3 |
| 50 | `mlx-fast-attention` | Using the Fused Attention Kernel | easy 3 | numeric | mlx | `causal-self-attention` | M3 |
| 60 | `mlx-weights-load` | Loading Weights into a Module Tree | medium 2 | numeric | mlx | `safetensors-parse` | M3 |
| 70 | `mlx-compile-step` | Compiling a Decode Step | medium 1 | numeric, cost | mlx | `mlx-lazy-eval-placement` | M3 |
| 80 | `mlx-compile-shapes` | Compiling Without Retracing | hard 1 | numeric, cost | mlx | `mlx-compile-step` | M3 |
| 90 | `mlx-preallocated-kv-cache` | A Step-Allocated KV Cache | easy 3 | numeric, cost | mlx | `preallocated-kv-cache` | M3 |
| 100 | `mlx-quantized-linear` | A Quantized Linear Layer | easy 3 | numeric, cost | mlx | `groupwise-4bit`, `quantized-matmul` | M3 |
| 110 | `mlx-quantize-module-tree` | Quantizing a Model with a Mixed Recipe | medium 2 | numeric, cost | mlx | `mlx-quantized-linear` | M3 |
| 120 | `mlx-streaming-generate` | Streaming Generation with Async Evaluation | medium 1 | numeric, cost | mlx | `mlx-lazy-eval-placement` | M3 |
| 130 | `mlx-benchmark-harness` | Benchmarking a Lazy Framework | medium 1 | numeric, cost | mlx | `mlx-lazy-eval-placement`, `latency-metrics` | M3 |
| 140 | `mlx-metal-kernel-elementwise` | A Custom Metal Kernel | medium 2 | numeric | mlx | - | M3 |
| 150 | `mlx-metal-kernel-dequant` | A Metal Kernel That Unpacks and Dequantizes | hard 1 | numeric, speed | mlx | `mlx-metal-kernel-elementwise`, `bit-packing` | M3 |
| 160 | `mlx-batched-generate` | Batched Generation with Ragged Prompts | hard 1 | numeric, cost | mlx | `batched-cached-decode`, `padded-batch-invariance` | M3 |
| 170 | `mlx-real-model-forward` | A Real Model in mlx.nn | hard 1 | numeric | mlx | `real-model-logits`, `mlx-weights-load` | M3 |
| 180 | `mlx-real-8bit-generate` | Running a 3B Model at 8 Bits | hard 2 | numeric, cost | mlx | `mlx-real-model-forward`, `mlx-quantized-linear`, `mlx-streaming-generate` | M3 |

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
| 10 | `cuda-inference-mode` | Inference Mode | medium 1 | numeric, property | torch | - | M4 |
| 20 | `cuda-device-placement` | Keeping Tensors on One Device | easy 3 | numeric, property | torch | - | M4 |
| 30 | `cuda-timing-sync` | Timing Asynchronous Work | easy 2 | numeric, cost | torch | `latency-metrics` | M4 |
| 40 | `cuda-sdpa-attention` | Fused Attention with the Right Mask | easy 3 | numeric | torch | `causal-self-attention` | M4 |
| 50 | `cuda-sync-free-decode-loop` | A Decode Loop Without Host Syncs | medium 2 | numeric, cost | torch | `cached-greedy-decode`, `categorical-sample` | M4 |
| 60 | `cuda-autocast-inference` | Mixed-Precision Inference | easy 2 | numeric, property | torch | `stable-softmax` | M4 |
| 70 | `cuda-pinned-transfer` | Pinned Memory and Non-Blocking Transfers | medium 1 | numeric, cost | torch | `cuda-device-placement` | M4 |
| 80 | `cuda-memory-accounting` | Reading the CUDA Allocator | easy 3 | numeric | torch | `kv-cache-bytes` | M4 |
| 90 | `cuda-oom-backoff` | Recovering from Out-of-Memory | medium 2 | numeric, cost | torch | - | M4 |
| 100 | `cuda-static-kv-cache` | A Static KV Cache | medium 1 | numeric, property | torch | `preallocated-kv-cache` | M4 |
| 110 | `cuda-compile-once` | Compiling a Decode Step Once | medium 1 | numeric, cost | torch | `cuda-static-kv-cache` | M4 |
| 120 | `cuda-graph-replay` | Capturing a CUDA Graph | medium 1 | numeric, cost | torch | `cuda-static-kv-cache` | M4 |
| 130 | `cuda-triton-rmsnorm` | A Triton Kernel for RMSNorm | medium 2 | numeric, property | torch | `rmsnorm` | M4 |
| 140 | `cuda-triton-dequant-linear` | A Fused Dequantize-and-Multiply Kernel | hard 1 | numeric, property | torch | `cuda-triton-rmsnorm`, `quantized-matmul` | M4 |

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
| 10 | `dynamic-batcher` | Dynamic Batching Policy | easy 3 | simclock | python | - | M1 |
| 20 | `sse-token-stream` | Streaming Tokens over Server-Sent Events | hard 1 | service | python | - | M1 |
| 30 | `token-bucket-limiter` | Token-Based Rate Limiting | medium 3 | simclock | python | - | M2 |
| 40 | `admission-control` | Admission Control and Backpressure | easy 3 | simclock | python | - | M2 |
| 50 | `metrics-endpoint` | A Metrics Endpoint | medium 3 | service | python | `latency-metrics` | M2 |
| 60 | `continuous-batching-scheduler` | Continuous Batching | medium 1 | simclock | python | `dynamic-batcher` | M2 |
| 70 | `priority-scheduler` | Priorities Without Starvation | easy 2 | simclock | python | `continuous-batching-scheduler` | M2 |
| 80 | `deadline-cancellation` | Deadlines and Cancellation | medium 2 | simclock | python | `continuous-batching-scheduler` | M2 |
| 90 | `chunked-prefill-scheduling` | Chunked Prefill Under a Token Budget | easy 2 | simclock | python | `continuous-batching-scheduler` | M2 |
| 100 | `vtc-fair-scheduler` | Fair Scheduling with Virtual Token Counters | medium 1 | simclock | python | `continuous-batching-scheduler` | M2 |
| 110 | `openai-chat-endpoint` | An OpenAI-Compatible Chat Endpoint | hard 1 | service | python | `sse-token-stream` | M2 |
| 120 | `least-loaded-balancer` | Power-of-Two-Choices Load Balancing | easy 2 | simclock, statistical | python | - | M5 |
| 130 | `retry-backoff-client` | A Client That Backs Off | medium 3 | simclock | python | `admission-control` | M5 |
| 140 | `graceful-drain` | Graceful Drain | medium 3 | simclock | python | `deadline-cancellation` | M5 |
| 150 | `streaming-backpressure` | Slow Consumers | hard 1 | simclock | python | `continuous-batching-scheduler` | M5 |
| 160 | `open-loop-load-generator` | An Open-Loop Load Generator | medium 3 | simclock, statistical | python | `latency-metrics` | M5 |
| 170 | `slo-deadline-scheduler` | Earliest-Deadline-First with Shedding | easy 3 | simclock | python | `priority-scheduler` | M5 |
| 180 | `multi-lora-batching` | Batching Requests Across Adapters | medium 1 | numeric, speed | t/j/m | `lora-merge` | M5 |
| 190 | `prefix-aware-router` | Prefix-Cache-Aware Routing | easy 3 | simclock | python | `prefix-cache-reuse`, `least-loaded-balancer` | M5 |
| 200 | `autoscaler-policy` | An Autoscaling Policy | easy 3 | simclock | python | `capacity-planner` | M5 |
| 210 | `preemption-recompute` | Preemption by Recompute | hard 1 | simclock | python | `continuous-batching-scheduler`, `paged-kv-allocator` | M5 |
| 220 | `prefill-decode-disaggregation` | Disaggregated Prefill and Decode | easy 3 | simclock | python | `chunked-prefill-scheduling` | M5 |

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
| 10 | `device-map-solver` | Solving a Device Map | medium 3 | numeric, property | python | `param-count` | M5 |
| 20 | `mmap-lazy-loader` | Lazy Weight Loading with mmap | medium 1 | numeric, cost | python | `safetensors-parse` | M5 |
| 30 | `embedding-pooling` | Pooling and Normalizing Embeddings | easy 3 | numeric, property | t/j/m | `attention-mask-builder` | M5 |
| 40 | `vector-topk-search` | Exact Vector Search | easy 3 | numeric, speed | t/j/m | `embedding-pooling` | M5 |
| 50 | `binary-rescore-search` | Binary Embeddings with Rescoring | medium 1 | numeric, cost | t/j/m | `vector-topk-search` | M5 |
| 60 | `ivf-index-search` | An Inverted-File Index | medium 2 | numeric, cost, property | t/j/m | `vector-topk-search` | M5 |
| 70 | `rerank-batching` | Batching a Reranker | medium 1 | numeric, cost | python | `pad-batch-left` | M5 |
| 80 | `semantic-cache` | A Semantic Response Cache | medium 3 | numeric, simclock | python | `vector-topk-search` | M5 |
| 90 | `cascade-router` | Routing Through a Model Cascade | easy 3 | numeric, cost | python | - | M5 |
| 100 | `tool-call-loop` | A Tool-Calling Loop | medium 2 | numeric, cost | python | `tool-call-stream-parser` | M5 |
| 110 | `tensor-parallel-linear` | Tensor-Parallel Layers on Simulated Devices | easy 3 | numeric, cost | t/j/m | `swiglu-mlp` | M5 |
| 120 | `pipeline-parallel-schedule` | Pipeline Parallelism Across Devices | hard 2 | simclock | python | `device-map-solver` | M5 |

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
| 10 | `reasoning-budget-processor` | Enforcing a Thinking-Token Budget | medium 2 | numeric, cost | t/j/m | `constrained-greedy-decode` | M5 |
| 20 | `cross-layer-kv-sharing` | Sharing KV Across Layers | medium 1 | numeric, cost | t/j/m | `cached-greedy-decode` | M5 |
| 30 | `diffusion-unmask-step` | One Step of Masked-Diffusion Decoding | medium 2 | numeric, statistical | t/j/m | `categorical-sample` | M5 |
| 40 | `kv-tiered-offload` | A Tiered KV Store | hard 2 | simclock, cost | python | `block-hash-prefix-cache` | M5 |
| 50 | `hybrid-state-checkpoints` | Prefix Reuse for Recurrent-State Models | medium 3 | numeric, cost | t/j/m | `prefix-cache-reuse` | M5 |
| 60 | `mla-latent-kv-cache` | Caching Latents with Multi-Head Latent Attention | hard 1 | numeric, property | t/j/m | `kv-cache-append`, `rope` | M5 |
| 70 | `speculative-tool-calls` | Speculative Tool Calls | hard 2 | simclock | python | `tool-call-loop`, `tool-call-stream-parser` | M5 |
| 80 | `expert-parallel-dispatch` | Expert Parallelism for MoE | hard 2 | numeric, cost | t/j/m | `moe-sparse-dispatch`, `tensor-parallel-linear` | M5 |

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
| 10 | `capstone-continuous-batching-engine` | A Continuous-Batching Engine with a Paged KV Cache | hard 3 | simclock, cost, property | python | `continuous-batching-scheduler`, `paged-kv-allocator`, `chunked-prefill-scheduling`, `preemption-recompute` | M5 |
| 20 | `capstone-quantized-runner` | A Quantized Model Runner | hard 3 | numeric, property, cost | t/j/m | `groupwise-4bit`, `quantized-matmul`, `decoder-forward`, `kv-cache-append` | M5 |
| 30 | `capstone-speculative-engine` | A Speculative Decoding Engine | hard 3 | statistical, numeric, cost | t/j/m | `speculative-sampling-block`, `speculative-decode-loop`, `dynamic-draft-length` | M5 |
| 40 | `capstone-structured-output-server` | A Structured-Output Server | hard 3 | service, numeric | python | `openai-chat-endpoint`, `json-schema-regex`, `dfa-token-index`, `tool-call-stream-parser` | M5 |
| 50 | `capstone-real-model-runtime` | A Complete Runtime for a Real Model | hard 3 | numeric, cost | t/j/m | `real-model-logits`, `real-model-cached-generate`, `tokenizer-json-load`, `sampler-chain` | M5 |
| 60 | `capstone-inference-gateway` | An Inference Gateway | hard 1 | simclock | python | `prefix-aware-router`, `admission-control`, `vtc-fair-scheduler`, `autoscaler-policy` | M5 |

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

## 6. Inference 75 (origin of the core lists)

This table is the origin of the core lists now kept in each `track.toml` (`core`): the 75
problems to do, in order, to cover what a strong inference engineer must be able to write
from memory. All 75 existed as of milestone 5. The core has since grown to 84 by adding
prerequisites. The table is kept as a historical record; the `core` lists are the source
of truth.

| # | stage | one sentence | problems |
|---|---|---|---|
| 1 | Numerics and tensors | Stable arithmetic and the indexing that every later problem assumes. | `stable-softmax`, `rmsnorm`, `attention-mask-builder`, `last-token-gather`, `sequence-log-likelihood` |
| 2 | Text to tokens and back | Encode, decode, and template text exactly as a production tokenizer does. | `bpe-encode`, `special-tokens-split`, `streaming-detokenizer`, `chat-template-render`, `tokenizer-json-load` |
| 3 | The forward pass | Build a current decoder-only model piece by piece, then match a real one. | `causal-self-attention`, `rope`, `swiglu-mlp`, `grouped-query-attention`, `decoder-forward`, `padded-batch-invariance`, `moe-router-forward`, `real-model-logits` |
| 4 | Choosing the next token | The samplers engines ship, composed in the right order, and when to stop. | `temperature-scaling`, `top-k-filtering`, `top-p-filtering`, `min-p-filtering`, `repetition-penalty`, `categorical-sample`, `sampler-chain`, `stop-sequences` |
| 5 | The KV cache | Compute each token once, then share, page, and index the memory. | `kv-cache-append`, `cached-greedy-decode`, `cached-rope-offsets`, `preallocated-kv-cache`, `batched-cached-decode`, `prefix-cache-reuse`, `paged-kv-allocator`, `radix-prefix-cache` |
| 6 | Structured output | Turn a regex or schema into per-step token masks and decode under them. | `token-mask-apply`, `dfa-token-mask`, `dfa-token-index`, `constrained-greedy-decode`, `json-schema-regex` |
| 7 | Quantization | Shrink weights, compute on the codes, and measure what was lost. | `affine-int8`, `symmetric-per-channel`, `groupwise-4bit`, `quantized-matmul`, `lora-merge`, `gptq-column-update` |
| 8 | Speculative decoding | Draft cheaply and verify exactly, first for one token, then in a loop. | `greedy-speculative-verify`, `speculative-sampling-step`, `speculative-sampling-block`, `speculative-decode-loop` |
| 9 | Sizing and measurement | Predict memory and speed from a config, and measure latency correctly. | `param-count`, `kv-cache-bytes`, `memory-bandwidth-bound`, `latency-metrics`, `latency-percentiles`, `goodput-slo` |
| 10 | Serving | Batch, admit, schedule, cancel, and stream under load. | `dynamic-batcher`, `sse-token-stream`, `token-bucket-limiter`, `admission-control`, `continuous-batching-scheduler`, `chunked-prefill-scheduling`, `deadline-cancellation`, `vtc-fair-scheduler`, `openai-chat-endpoint`, `preemption-recompute`, `prefix-aware-router` |
| 11 | On the GPU | Lazy evaluation, compilation, and static caches on MLX and CUDA. | `mlx-lazy-eval-placement`, `mlx-compile-step`, `mlx-streaming-generate`, `cuda-sync-free-decode-loop`, `cuda-static-kv-cache` |
| 12 | Around the model | Place a model across devices and drive it from an application loop. | `device-map-solver`, `tool-call-loop` |
| 13 | Capstones | Put it together: an engine and a complete runtime for a real model. | `capstone-continuous-batching-engine`, `capstone-real-model-runtime` |

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

## 9. Modern architectural components (2026-11-21)

**Status: built.** The owner approved this section as proposed on 2026-11-21, and all 24
problems, the kit additions, the five tracks, and the area `model-architectures` exist
and are committed. Each batch was authored, reviewed by a second subagent, fixed, and
re-validated ([`PROBLEM_PROCESS.md`](PROBLEM_PROCESS.md), step 6). The text below is the
proposal as written, kept as the record of what was specified and why; where the built
problem differs, its spec paragraph ends with an "As built" sentence, and the built
problem is the authority. Section numbers 1 to 8 are unchanged. Written on 2026-11-21 as steps 1 and 2 of
[`PROBLEM_PROCESS.md`](PROBLEM_PROCESS.md) for the owner's goal: problems on "modern LLM
architectural components, just as thoroughly", from "the top open-weights models as-of
Nov 2026", with no training topics and no image or vision-language models.

How the mechanics were established. The starting inventory (the 2026-11-20 survey) read
configs and model cards only. For this proposal the mechanics were read in the
reference implementations installed in this repository's environment, `transformers`
5.23.0 and `mlx-lm` 0.32.2 (paths are in [Sources](#sources)), in each shipped model's
`config.json` fetched on 2026-11-21, and, where noted, in the paper. Every spec below
names what it follows. Where the paper and the implementation differ, both are recorded
and the problem follows the implementation. Three restated formulas (the chunked delta
rule, the state-space matrix form, the Sinkhorn claims behind two mutants) were also
checked numerically with a throwaway script. Labels: **VERIFIED** means read in a
primary source on 2026-11-21; **UNVERIFIED** means not read, and nothing unverified is
specified as a problem.

### 9.1 Area and modules

A new area of five modules (tracks), 24 problems. The owner accepted the ids and the
titles as proposed. As built: the area's description in `areas/model-architectures.toml`
is reworded, and each module has a scene of a second series (the sea) and a color.

Area `model-architectures`, working title "Model architectures", one sentence: "The
components that distinguish the open-weights models of 2026 from a plain decoder:
recurrent and sparse mixers, their caches, and the heads and streams around them."

| order | module id | working title | one sentence | problems | core, in teaching order |
|---|---|---|---|---|---|
| 210 | `recurrent-mixers` | Recurrent and Hybrid Mixers | Layers that carry a fixed-size state instead of a growing cache: the gated delta rule, a state-space step, a gated convolution, and a decoder that mixes them with attention. | 6 | `gated-delta-step`, `gated-delta-layer`, `mamba2-ssd-step`, `hybrid-decoder-forward` |
| 220 | `sparse-attention` | Indexed and Compressed Attention | Attend to a chosen few: score the past with a cheap indexer, attend over the top-k only, and keep the indexer's own cache correct while decoding. | 5 | `indexer-topk-select`, `gathered-sparse-attention`, `indexer-cache-decode` |
| 230 | `block-components` | Positions, Experts, and Streams | The parts of a block that changed since 2024: partial and scaled rotary positions, sigmoid-routed experts with a shared expert, per-layer embeddings, and multi-stream residuals. | 7 | `partial-rope`, `yarn-frequencies`, `sigmoid-router-shared-expert`, `per-layer-embeddings` |
| 240 | `drafts-and-diffusion` | Built-in Drafts and Text Diffusion | Output paths beyond one token per pass: the chained multi-token-prediction module that ships inside checkpoints, and block text diffusion over a cached prefix. | 3 | `mtp-chain-draft`, `entropy-bound-denoise-step`, `block-diffusion-decode` |
| 250 | `real-hybrid-model` | A Real Hybrid Model | Read a real hybrid model's configuration, then match one of its recurrent blocks and its logits from the raw weights. | 3 | `layer-plan-from-config`, `real-gdn-block`, `real-hybrid-logits` |

No core problem has a non-core prerequisite, in its own module or in an existing one
(script-checked against the `core` lists in `problems/*/track.toml`). Difficulty: 11
easy, 11 medium, 2 hard. Two problems need real weights (`real-gdn-block`,
`real-hybrid-logits`); the other 22 use small seeded tensors on a CPU in float32.

### 9.2 What the sources say (component by component)

All rows VERIFIED on 2026-11-21 unless the last column says otherwise. "tf" is
`transformers` 5.23.0, "mlx" is `mlx-lm` 0.32.2.

| component | what is computed (as shipped) | read in | shipped in (config read) | paper, and where it differs |
|---|---|---|---|---|
| Gated delta rule | Per value head, state `S` of shape `(Dv, Dk)`. Per token: `S = g * S`; `S = S + beta * (v - S k) k^T`; output `y = S q`, read after the write. `g = exp(-exp(A_log) * softplus(a + dt_bias))`, `beta = sigmoid(b)`, with `a`, `b` linear projections of the layer input. `q` and `k` are L2-normalized as `x * rsqrt(sum(x^2) + 1e-6)` and `q` is also multiplied by `Dk^-0.5`. Key heads are repeated (`repeat_interleave`) to value heads. | tf `qwen3_5/modeling_qwen3_5.py` (`torch_recurrent_gated_delta_rule`, `Qwen3_5GatedDeltaNet`); mlx `gated_delta.py`, `qwen3_5.py` | `Qwen/Qwen3.5-0.8B`, `Qwen3.5-2B` (16 key and 16 value heads of 128, conv kernel 4) | Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv 2412.06464, ICLR 2025), Eq. 10: `S_t = S_{t-1}(alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T`, `o_t = S_t q_t`. Same recurrence. The paper leaves the `alpha` parameterization to a footnote ("Mamba2's"); the `Dk^-0.5` on `q` is in the code, not in the text read. tf stores the state transposed, `(Dk, Dv)`. |
| Gated delta layer | `in_proj_qkv(x)`, a causal depthwise convolution (kernel 4, no bias) over the concatenation of the saved last 3 inputs and the new ones, SiLU, split into `q`, `k`, `v`; the recurrence; then a gated norm per value head: `weight * rms_norm(y) * silu(z)` with `z = in_proj_z(x)` (norm first, then gate); `out_proj`. State per layer: conv state `(K-1, 2*key_dim + value_dim)` and `S`. | same files (`Qwen3_5RMSNormGated`, `causal_conv1d_update`) | same | Paper figure: projection, short convolution, SiLU, L2 norm; "normalization and gating before applying output projection". Same order. |
| Chunked delta rule | For a chunk with cumulative log-decay `c_i`, `G_ij = exp(c_i - c_j)` for `i >= j`: solve the unit-lower-triangular systems `(I + tril((K_b K^T) * G, -1)) U = V_b` and `... W = K_b * exp(c)`, where `K_b = beta K`, `V_b = beta V`; `V' = U - W S0^T`; `Y = (Q * exp(c)) S0^T + tril((Q K^T) * G) V'`; `S1 = S0 * exp(c_last) + V'^T (K * exp(c_last - c))`. Padding rows are zero with log-decay 0 and `beta` 0. | tf `torch_chunk_gated_delta_rule` (chunk size 64) | used by tf for every multi-token call; the step form only for one cached token | Paper section 3.3 (chunkwise form with the WY representation and UT transform). The restatement here was checked numerically against the step form (largest difference 4e-16 in float64). |
| Mamba-2 step | Per head, state `h` `(P, N)`: `dt = softplus(dt_raw + dt_bias)`; `h = h * exp(dt * A) + dt * x B^T` with `A = -exp(A_log)`; `y = h C + D x`. `B`, `C` have `n_groups` groups, each serving `H / n_groups` consecutive heads. The block gates before it normalizes: `rms_norm(y * silu(gate))`, in groups. | tf `nemotron_h/modeling_nemotron_h.py` (`mamba2_selective_state_update`, `NemotronHRMSNormGated`); mlx `ssm.py`, `mamba2.py` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (64 heads of 64, state 128, 8 groups, conv 4; blocks alternate `mamba`, `moe`, and 6 `attention`) | Dao and Gu, "Transformers are SSMs" (arXiv 2405.21060, ICML 2024), abstract only read. Implementations differ on the `dt` clamp: mlx clips `dt` to `time_step_limit` always; tf clamps to `(time_step_min, inf)` in the chunked path and not in the one-token path. |
| Gated convolution block | `B, C, h = chunk3(in_proj(x))`; `y = out_proj(C * conv(B * h))` with a causal depthwise convolution of kernel `conv_L_cache` and no activation. With `use_sandwich_norm` the block output is normalized again before the residual add. | tf `lfm3/modeling_lfm3.py` (`Lfm3ShortConv`) | `LiquidAI/LFM3-1.7B` (kernel 7, 13 of 24 layers) | No paper read. License of LFM3 not checked; no weights are used. |
| Hybrid schedule and gated attention | Layer `i` is full attention when `(i + 1) % full_attention_interval == 0`, else linear attention; an explicit `layer_types` list wins. Full attention: `q_proj` outputs `2 * head_dim` per head, the first half the query and the second half a gate; `attn_output * sigmoid(gate)` before `o_proj`; RMS QK-norm; partial RoPE. RMSNorm weights are stored zero-centered (`x_normed * (1 + weight)`) for the block norms, the final norm, `q_norm`, and `k_norm`, but not for the gated norm inside the delta layer. | tf `Qwen3_5Attention`, `Qwen3_5RMSNorm`, `Qwen3_5TextConfig`; mlx `qwen3_5.py` (`sanitize` adds 1 to those weights) | `Qwen/Qwen3.5-0.8B`: 24 layers, layers 3, 7, ..., 23 full attention | Model family has no architecture paper that was read. |
| Indexer scoring and top-k (DSA) | `I[t, s] = sum_h w[t, h] * relu(q_I[t, h] . k_I[s])`, with `q_I` projected from the query's low-rank residual, one shared key `k_I[s] = LayerNorm(wk(x_s))`, `w = weights_proj(x) * H^-0.5`, and `D^-0.5` on the dot product. Future tokens are masked to `-inf`, then `topk(min(k, S))`; the selected set is intersected with the causal mask. Indexer keys are cached per layer. | tf `glm_moe_dsa/modeling_glm_moe_dsa.py` (`GlmMoeDsaIndexer`, `GlmMoeDsaAttention`) | `zai-org/GLM-5.3` (32 indexer heads of 128, top-k 2048), `moonshotai/Kimi-K3.1-Pro` (`dsa_config`, config only) | "DeepSeek-V3.2" (arXiv 2512.02556), Eq. 1 and 2: the same score and "ReLU ... for throughput". The scale factors, the LayerNorm, and RoPE on the first `qk_rope_head_dim` indexer channels are from the code. |
| Shared indexers across layers | `indexer_types[i]` is `"full"` or `"shared"`; a shared layer attends with the top-k indices of the nearest preceding full layer, which are passed up the stack. Default rule: full when `max(i - offset + 1, 0) % freq == 0`. | tf `GlmMoeDsaConfig`, `GlmMoeDsaModel.forward` | `zai-org/GLM-5.3`: `index_topk_freq` 4, `index_skip_topk_offset` 3, giving 3 full layers then repeating 3 shared, 1 full | "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse" (arXiv 2603.12201), abstract only read. |
| Heavily compressed attention (HCA) | Each closed window of `rate` tokens becomes one key-value entry: `rms_norm(sum_j softmax_j(Z_j + position_bias_j) * C_j)`, softmax over the window per channel, RoPE at position `w * rate`. Tokens of an unfinished window wait in a buffer across calls. Query at position `t` may see entry `w` only when `w < (t + 1) // rate`. | tf `deepseek_v4/modeling_deepseek_v4.py` (`DeepseekV4HCACompressor`, `DeepseekV4HCACache`) | `deepseek-ai/DeepSeek-V4-Pro-0813` (`compress_ratios` alternate 128 and 4, window 128) | "DeepSeek-V4" (arXiv 2606.19348), abstract only read; the code cites section 2.3.2, Eq. 20 to 23. The V4.1 checkpoint (`deepseek_v41`) has no tf model; its mlx file was not read. |
| Partial RoPE, two conventions | Default with `partial_rotary_factor f`: `r = int(d * f)`, `inv_freq_i = theta^(-2i / r)`, rotate the first `r` channels (half-split inside them), pass the rest. Proportional (`rope_type: "proportional"`): `a = int(f * d // 2)` rotated pairs with `inv_freq_i = theta^(-2i / d)` (denominator the full head dimension), zero frequency for the other pairs, half-split over the whole head. | tf `Qwen3_5TextRotaryEmbedding`, `apply_rotary_pos_emb`; `modeling_rope_utils.py` (`_compute_proportional_rope_parameters`) | default: Qwen3.5 (0.25, theta 1e7), LFM3 full layers (0.5, theta 1e6); proportional: `google/gemma-4-E2B-it` full layers (0.25, theta 1e6, head dim 512; sliding layers default, theta 1e4, head dim 256) | Gemma 4 report (arXiv 2607.02770) not read. |
| YaRN | `low`, `high = dim * ln(L0 / (beta * 2 pi)) / (2 ln base)` for `beta_fast` 32 and `beta_slow` 1, floored and ceiled when `truncate`, clamped to `[0, dim - 1]`; `ramp_i = clamp((i - low) / (high - low), 0, 1)`; `inv_freq = (1 - ramp) / pos_freq + ramp / (factor * pos_freq)`; `attention_factor = 0.1 ln(factor) + 1` multiplies the cos and sin tables. | tf `modeling_rope_utils.py` (`_compute_yarn_parameters`) | `openai/gpt-oss-20b` (factor 32, original 4096, theta 150000, `truncate: false`), DeepSeek-V4 (factor 16, original 65536) | Peng et al., "YaRN" (arXiv 2309.00071), title page only read. With `mscale_all_dim` set, GLM folds the factor into the softmax scale instead. |
| Shared key-value with output rotation | One KV head whose tensor is both key and value; RoPE on the trailing `rope_dim` channels in interleaved pairs; after attention the output's rope slice is rotated by minus the query position, so each contribution depends on relative distance. | tf `DeepseekV4Attention`, `apply_rotary_pos_emb` | DeepSeek-V4 (`num_key_value_heads` 1, head dim 512, `qk_rope_head_dim` 64) | Paper body not read. |
| Sigmoid-family router with a shared expert | `scores = score_fn(x W^T)`, `score_fn` sigmoid or `sqrt(softplus(.))`; top-k chosen by `scores + e_score_correction_bias`; weights are the unbiased scores of the chosen experts, divided by `(sum + 1e-20)`, then multiplied by `routed_scaling_factor`; the shared expert's output is added unweighted. | tf `GlmMoeDsaTopkRouter`, `GlmMoeDsaMoE`, `DeepseekV4TopKRouter`, `activations.py` (`SqrtSoftplusActivation`) | GLM-5.3 (256 experts, top-8, 1 shared, scale 2.5, first 3 layers dense), Nemotron 3.5, Kimi K3.1 (sigmoid); DeepSeek-V4 (`sqrtsoftplus`) | "DeepSeek-V3 Technical Report" (arXiv 2412.19437), Eq. 12 to 16: sigmoid affinity, normalization among the selected, "the bias term is only used for routing". The scaling factor is in the code, not in those equations. All configs read have one expert group. |
| Latent experts | Router on the full-width input; `down_proj` to the latent width; experts and the weighted sum at latent width; optional RMSNorm at latent width; `up_proj`; the shared expert runs at full width on the original input and is added last. | mlx `kimi_k3.py` (`KimiK3SparseMoE`); tf `NemotronHMoE` (same without the norm) | `moonshotai/Kimi-K3.1-Pro` (`routed_expert_hidden_size` 3584 of 7168, `latent_moe_use_norm` true). Nemotron 3.5 has `moe_latent_size` null | No paper read. Moonshot's own modeling file was not read; mlx implements model type `kimi_k3`, the config's is `kimi_k31`. |
| Per-layer embeddings | Token part: a packed table of width `L * d_ple`, scaled by `sqrt(d_ple)`, reshaped `(T, L, d_ple)`. Context part: a projection of the input embeddings times `hidden^-0.5`, reshaped, RMS-normalized. Sum times `2^-0.5`. In each layer, after the MLP residual: `h + norm(proj(act(gate(h)) * ple[:, i]))`. | tf `gemma4/modeling_gemma4.py` (`get_per_layer_inputs`, `project_per_layer_inputs`, `Gemma4TextDecoderLayer`) | `google/gemma-4-E2B-it` (`hidden_size_per_layer_input` 256, 35 layers, 4 sliding to 1 full, 20 KV-shared layers) | Gemma 4 report not read. The survey's "5 to 1" and "18 shared" are other sizes or slips. |
| Hyper-connections (mHC) | `N` residual streams. Per sublayer: flatten, unweighted RMS norm, one linear map to `pre` (`N`), `post` (`N`), `comb` (`N x N`); `pre = sigmoid(.) + eps`, `post = 2 sigmoid(.)`, `comb = softmax(., -1) + eps` then Sinkhorn (columns first, 20 iterations); sublayer input `sum_n pre_n x_n`; output streams `post_n * f(.) + (comb^T X)_n`. Embeddings are copied to all streams; a final head collapses them. | tf `DeepseekV4HyperConnection`, `DeepseekV4DecoderLayer`, `DeepseekV4HyperHead` | DeepSeek-V4 (`hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6) | Code cites "Xie et al., 2026" and paper section 2.2; neither read. |
| Chained multi-token prediction | Stage `k`: `x = eh_proj([enorm(Emb(t_{i+k+1})); hnorm(h^{k})])`, one decoder block, a post norm, the shared head; embedding and head shared with the main model; stage `k + 1` reads stage `k`'s output and the token stage `k` drafted. `h^0` is the main model's final-normed hidden state. | tf `modeling_layers.py` (`MtpLayer`, `MtpModel`), `generation/candidate_generator.py` (`MTPCandidateGenerator`), `utils/output_capturing.py` | weights `mtp.*` in `Qwen/Qwen3.5-0.8B` (`mtp_num_hidden_layers` 1; names read in `model.safetensors.index.json`), GLM-5.3 and Nemotron 3.5 (1 layer), DeepSeek-V4.1 (3) | DeepSeek-V3 report, Eq. 21 to 23: `h' = M_k[RMSNorm(h^{k-1}); RMSNorm(Emb(t_{i+k}))]`, hidden state first. tf concatenates the embedding first unless `mtp_hidden_states_first`. The problem follows tf. |
| Block text diffusion | Canvases of `canvas_length` tokens start as uniform random tokens. Each step: denoiser logits (conditioned on the cached prefix and on the previous step's logits), temperature `t_min + (t_max - t_min) * n / N` with `n` steps remaining, sample a canvas, accept the lowest-entropy positions while `cumsum(H) - H <= entropy_bound`, renoise every other position with new random tokens. Stop when the argmax canvas is unchanged for `stability_threshold` steps and the mean entropy is below `confidence_threshold`. The argmax canvas is appended and encoded into the prefix cache once. | tf `diffusion_gemma/generation_diffusion_gemma.py` (`EntropyBoundSampler`, `StableAndConfidentStoppingCriteria`, `generate`) | `google/diffusiongemma-26B-A4B-it` (`canvas_length` 256) | "Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking" (arXiv 2505.24857), abstract only: written for masked diffusion, where an unmasked token stays. The shipped sampler is for uniform noise and recomputes acceptance every step, so an accepted position can be renoised later. |

### 9.3 The real model

`Qwen/Qwen3.5-0.8B`, the smallest shipped hybrid of gated delta layers and attention.
Nothing was downloaded except `config.json` and `model.safetensors.index.json`.

| fact | value | status |
|---|---|---|
| repository, revision | `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`, created 2026-02-28, ungated | VERIFIED (Hugging Face API) |
| license | Apache-2.0 (card metadata; the `LICENSE` file of 11,544 bytes is the Apache License 2.0 text with "Copyright 2026 Alibaba Cloud") | VERIFIED (metadata; file text read 2026-11-21) |
| parameters | 873,438,784 in the file (873,436,192 BF16 and 2,592 F32), which includes a 12-layer vision tower (153 tensors) and the MTP module (15 tensors); the text decoder is 320 tensors, 752,393,024 parameters | VERIFIED (API, index; the text decoder's count read from the safetensors header 2026-11-21) |
| files | `model.safetensors-00001-of-00001.safetensors` 1,746,942,600 bytes; `config.json` 2,907; `tokenizer.json` 12.8 MB; `vocab.json`, `merges.txt`, `chat_template.jinja` | VERIFIED |
| text decoder | 24 layers, hidden 1024, SwiGLU 3584, vocabulary 248,320, tied embeddings (no `lm_head` tensor), RMSNorm eps 1e-6 | VERIFIED (config, index) |
| schedule | `layer_types`: linear attention except layers 3, 7, 11, 15, 19, 23 (18 delta layers, 6 attention layers) | VERIFIED |
| delta layers | 16 key heads and 16 value heads of 128; conv kernel 4 over 6144 channels | VERIFIED |
| attention layers | 8 heads, 2 KV heads, head dim 256; `q_proj` is `(4096, 1024)` (query and gate); RoPE theta 1e7 on the first 64 channels | VERIFIED (config; shapes read from the safetensors header 2026-11-21) |
| tensor names | `model.language_model.layers.{i}.linear_attn.{A_log, dt_bias, conv1d.weight, in_proj_qkv.weight, in_proj_z.weight, in_proj_a.weight, in_proj_b.weight, norm.weight, out_proj.weight}`, `...self_attn.{q,k,v,o}_proj.weight`, `...{q,k}_norm.weight`, `...mlp.{gate,up,down}_proj.weight`, `...input_layernorm.weight`, `...post_attention_layernorm.weight`, `model.language_model.norm.weight`, `model.language_model.embed_tokens.weight` | VERIFIED (index) |
| library support | `transformers` 5.23.0 (installed) has `models/qwen3_5` with `Qwen3_5ForCausalLM`, which ignores `mtp.*` and `model.visual.*`; `mlx-lm` 0.32.2 (installed) has `models/qwen3_5.py`. No newer version is needed | VERIFIED by reading the installed code and, 2026-11-21, by loading the model: `Qwen3_5ForCausalLM.from_pretrained` maps the `model.language_model.` prefix (no missing or unexpected keys; it needs `model.safetensors.index.json`), and `mlx_lm.utils.load_model` loads it |
| which tensors are F32, exact tensor shapes | F32: `linear_attn.A_log` `(16,)` and `linear_attn.norm.weight` `(128,)` of the 18 delta layers (36 tensors, 2,592 numbers); everything else BF16, `dt_bias` included. Shapes: `in_proj_qkv` `(6144, 1024)`, `in_proj_z` `(2048, 1024)`, `in_proj_a`, `in_proj_b` `(16, 1024)`, `conv1d` `(6144, 1, 4)`, `out_proj` `(1024, 2048)`; `q_proj` `(4096, 1024)`, `k_proj`, `v_proj` `(512, 1024)`, `o_proj` `(1024, 2048)`, `q_norm`, `k_norm` `(256,)`; `gate_proj`, `up_proj` `(3584, 1024)`, `down_proj` `(1024, 3584)`; norms `(1024,)`; `embed_tokens` `(248320, 1024)`; no `lm_head` | VERIFIED (safetensors header of the downloaded file, 2026-11-21) |

Why this model: it is the only checkpoint of 4B parameters or fewer found that ships
gated delta layers, the 3-to-1 schedule, gated attention, partial RoPE, and an MTP
module together, and it is permissively licensed with safetensors weights.
`Qwen/Qwen3.5-0.8B-Base` is the same size without a chat template; `Qwen/Qwen3.5-2B`
(2.27 B, 4.55 GB) is the same architecture at width 2048. Section 2 rejected the Qwen3.5
small models as house models because they are multimodal and mostly linear attention;
that is the reason to use one here, for these two problems only.

What verifying a from-scratch forward pass requires, in order: the owner's decision to
download 1.75 GB; an asset entry `qwen3.5-0.8b` pinned to the revision above; loading
with `Qwen3_5ForCausalLM.from_pretrained(..., dtype=float32)` and with `mlx-lm`, and
confirming they agree (the checkpoint's names start with `model.language_model.`, and
whether the text-only class maps that prefix was not tested); a `fixtures/make.py` that writes logits and per-layer hidden
states for prompts of at most 16 token ids (ids are fixtures, so no tokenizer work is
needed); a kit reference forward (`RealHybridLM`) written from the list in
`real-hybrid-logits` below and matched to those fixtures; then a measured tolerance.
`transformers` runs multi-token calls through the chunked delta rule in float32, so
agreement with a step-by-step solution is expected near 1e-4 but must be measured, not
assumed (section 7, item 8 of the process document).

Measured 2026-11-21, after the owner approved the download (asset `qwen3.5-0.8b`,
1.76 GB with `tokenizer.json` and the index; details and what was not verified in
`PROBLEM_AUTHORING.md` section 15, "The hybrid model"). A from-scratch forward pass in
float32 against `transformers` 5.23.0 on eight prompts of at most 16 ids, in numpy,
torch, JAX, and MLX: logits within 1.9e-4 (magnitude 12 to 31), the argmax equal at all
86 positions, hidden states after layers 0, 1, 12, and 13 within 5.3e-6 (their typical
entry is 0.02 to 0.04), one delta block run on the reference input within 3.0e-6.
Tolerances to use: `rtol=1e-3, atol=1e-3` for the logits, and `rtol=1e-3, atol=1e-5` for
a block's hidden state (an `atol` of 1e-3 is too loose at that scale). `mlx-lm` 0.32.2
does **not** agree with `transformers` on this checkpoint: it adds the `1.0` of the norm
weights in bfloat16 when it loads, and its logits differ by 0.70 as loaded and 0.17
after `set_dtype(float32)` (1.7e-4 when the weights are cast before that step), so
fixtures come from `transformers`. `transformers`' own chunked and cached paths differ
by up to 1.5e-4.

### 9.4 Kit additions needed

Interfaces are driven by the specs that use them and follow the kit rules of
`PROBLEM_AUTHORING.md` section 7.

| name | interface | first used by |
|---|---|---|
| `ops` additions | `ops.l2norm(x, eps=1e-6)` (`x * rsqrt(sum(x^2) + eps)`), `ops.causal_conv1d(x, w, state)` (depthwise, cross-correlation, returns output and the last `K - 1` inputs), `ops.partial_rope(x, positions, theta, factor)`, `ops.gated_delta_layer(x, params, state)`, `ops.eb_accept(logits, entropy_bound)` | `hybrid-decoder-forward`, `block-diffusion-decode` |
| `GDNConfig`, `gdn_params` | `gdn_params(cfg, seed) -> dict[str, ndarray]` with the checkpoint's names (`in_proj_qkv.weight`, `in_proj_z.weight`, `in_proj_a.weight`, `in_proj_b.weight`, `conv1d.weight` `(C, 1, K)`, `A_log`, `dt_bias`, `norm.weight`, `out_proj.weight`); `cfg` has `d_model`, `n_key_heads`, `n_value_heads`, `key_dim`, `value_dim`, `conv_kernel`. Oracles `gdn_step_ref`, `gdn_layer_ref(cfg, params, x, state)`. Gains and `A_log` drawn away from their defaults so each matters. | `gated-delta-layer` |
| `HybridConfig`, `hybrid_params`, `hybrid_forward_ref` | Extends `DecoderConfig` with `layer_types` (or `full_attention_interval`), the delta-layer sizes, `partial_rotary_factor`, and gated attention; `hybrid_params(cfg, seed)` uses `model.layers.{i}.linear_attn.*` or `self_attn.*` names; `hybrid_forward_ref(cfg, params, tokens, state=None) -> (logits, state)`. Norm weights are plain gains here (the zero-centered storage is taught on the real model). | `hybrid-decoder-forward` |
| `CountingKV` | `store = CountingKV(k, v)`; `store.gather(rows) -> (k_rows, v_rows)` for an integer array of any shape, counting `counters.rows_read`; `store.n`; `store.append(k, v) -> CountingKV`. | `gathered-sparse-attention`, `indexer-cache-decode` |
| `TinyDSA` | Seeded stack of sparse-attention layers: `dsa.index_key(layer, x)` (counts `index_key_rows`), `dsa.index_query(layer, x) -> (q_I, w)`, `dsa.qkv(layer, x)`, `dsa.out(layer, a)`, `dsa.indexer(layer, h) -> idx` (counts `indexer_calls`), `dsa.attend(layer, h, idx)`, `dsa.types`, and numpy oracles for the full causal pass. | `indexer-cache-decode`, `shared-index-layers` |
| `mtp_params`, `mtp_draft_ref` | `mtp_params(cfg, n_layers, seed)` adds `mtp.layers.{k}.{enorm.weight, hnorm.weight, eh_proj.weight, post_norm.weight}` and `mtp.layers.{k}.mtp_block.*` (one kit decoder block) to a `decoder_params` dict; the oracle returns drafted tokens and logits. | `mtp-chain-draft` |
| `BlockDenoiser` | Seeded model with `model.encode(tokens, cache) -> cache` (causal, counts `encoder_tokens` and `encoder_calls`) and `model.denoise(canvas, cache, self_cond_logits) -> logits` (bidirectional over the canvas, counts `denoise_calls`); `vocab_size`, `eos_id`, `pad_id`; a `sharpness` option so that some fixtures converge early and some never do. | `block-diffusion-decode` |
| asset `qwen3.5-0.8b`, `RealHybridLM` | `AssetSpec` in the kit registry; `RealModel` reads the nested `text_config` into a config with `layer_types` and the delta-layer sizes; `model.lm()` returns the reference forward with `block(x, layer, state)` and `hidden_states(tokens)`. Only after the owner approves the download. | `real-gdn-block`, `real-hybrid-logits` |

Nothing new is needed by `gated-delta-step`, `gated-delta-chunked`, `mamba2-ssd-step`,
`gated-conv-block`, `indexer-topk-select`, `compressed-kv-stream`, `partial-rope`,
`yarn-frequencies`, `shared-kv-rotation`, `sigmoid-router-shared-expert`, `latent-moe`
(existing `CountingExperts`), `per-layer-embeddings`, `hyper-connections`,
`entropy-bound-denoise-step`, and `layer-plan-from-config`: their inputs are plain
seeded tensors or dicts.

### 9.5 The proposed catalog

Milestone `M6` for every row. `requires` is `cpu`.

#### Module `recurrent-mixers` (order 210)

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `gated-delta-step` | One Step of the Gated Delta Rule | easy 2 | numeric, property | t/j/m | - | M6 |
| 20 | `gated-delta-layer` | A Gated DeltaNet Layer with Its State | medium 3 | numeric, property | t/j/m | `gated-delta-step` | M6 |
| 30 | `gated-delta-chunked` | The Delta Rule a Chunk at a Time | hard 1 | numeric, property, cost | t/j/m | `gated-delta-step` | M6 |
| 40 | `mamba2-ssd-step` | A State-Space Step and Its Attention Form | easy 3 | numeric, property | t/j/m | - | M6 |
| 50 | `gated-conv-block` | A Gated Convolution Block | easy 3 | numeric, property | t/j/m | - | M6 |
| 60 | `hybrid-decoder-forward` | A Hybrid Decoder: Three Recurrent Layers to One Attention Layer | medium 3 | numeric, property | t/j/m | `gated-delta-layer`, `kv-cache-append`, `decoder-forward` | M6 |

**`gated-delta-step`** — Implement `gdn_gates(a, b, A_log, dt_bias) -> (g, beta)` with `g = exp(-exp(A_log) * softplus(a + dt_bias))` and `beta = sigmoid(b)`, and `gdn_step(S, q, k, v, g, beta) -> (y, S)` for one token and `H` heads, `S` of shape `(H, Dv, Dk)`: decay `S = g * S`, write `S = S + beta * (v - S k) k^T`, read `y = S q` from the updated state (Gated DeltaNet, Yang et al., 2024, Eq. 10; the recurrent path of `transformers` 5.23 `qwen3_5` and of `mlx-lm` 0.32 `gated_delta.py`, which is the layout followed). Goal: a memory that overwrites what it stored under a key instead of adding to it. Numeric, plus the property that with `g = 1`, `beta = 1`, and a unit-norm key, `S k` equals `v` after the step whatever `S` was. Mutants: output read before the write; additive write `beta * v k^T` with no `S k` correction; decay applied after the write; the correction computed from the undecayed state; `softplus` missing from the gate (so `g` can exceed 1 for negative `a`). As built (2026-11-21): inputs, including `S`, are not modified, and a case checks it; both ends of the gate (`g = 0` and `g = 1`) are tested.

**`gated-delta-layer`** — Implement `gdn_layer(params, cfg, x, state) -> (out, state)` for `x` of shape `(T, D)` and `state = (conv_state, S)` or `None`: `in_proj_qkv`; a causal depthwise convolution over the saved last `K - 1` projected inputs followed by the new ones, then SiLU; split into `q`, `k`, `v`; L2-normalize `q` and `k` (`x * rsqrt(sum(x^2) + 1e-6)`) and multiply `q` by `Dk^-0.5`; repeat each key head over its `Hv / Hk` consecutive value heads; gates from `in_proj_a` and `in_proj_b`; the recurrence of `gated-delta-step`; `norm.weight * rms_norm(y) * silu(in_proj_z(x))` per value head; `out_proj` (the `linear_attn` sublayer of Qwen3.5 as in `transformers` 5.23 and `mlx-lm` 0.32). Goal: the two pieces of state a recurrent layer carries, and that neither can be rebuilt from the other. Property: any chunking of the sequence gives the output of one pass; numeric against `gdn_layer_ref` with `Hv = 2 Hk`. Mutants: conv state reset to zeros on each call; conv state taken from the convolution's output instead of its input; gate applied before the norm (the Mamba-2 order); key heads tiled instead of repeated; `Dk^-0.5` on `q` omitted; RMS norm over the whole value width instead of per head. [kit: `gdn_params`] As built (2026-11-21): the signature has a `cfg`, because the number of key heads cannot be read from the tensor shapes, and puts `params, cfg` first as `decoder-forward` does; the state passed in is not modified.

**`gated-delta-chunked`** — Implement `gdn_chunked(q, k, v, log_g, beta, memory, chunk_size) -> (Y, memory)` that equals a loop of `gated-delta-step` over `T` tokens but loops only over chunks: inside a chunk, with cumulative log-decay `c`, solve the two unit-lower-triangular systems of section 9.2, correct the new values by what `S0` already predicts, and carry `S` to the next chunk with the chunk's total decay (the chunkwise form of Yang et al., 2024, section 3.3, as written in `torch_chunk_gated_delta_rule` of `transformers` 5.23; the last partial chunk is padded with zero rows, log-decay 0, and `beta` 0). Goal: why a recurrent layer can still prefill in parallel. Property: equal to the step loop for every chunk size, a nonzero `S0`, and `T` not a multiple of the chunk size (`rtol` 1e-3); speed: within a measured factor of the reference at `T = 1024`, sized so that a per-token loop fails. Mutants: the per-token loop (slow); the chunk's decay not applied to the carried state; the diagonal excluded from the within-chunk read (a token does not see its own write); `S0` left out of the value correction; padding with decay 0 instead of log-decay 0, which erases the state. Authors: MLX `mx.linalg.solve_triangular` runs only with `stream=mx.cpu` (measured on MLX 0.32.4); a forward substitution over chunk positions is an accepted alternative. As built (2026-11-21): the speed judge is gone. An independent reviewer's per-token loop compiled with `jax.jit` and `lax.scan` ran faster than the reference and was accepted, and honest forward substitution sat within 2x of rejected variants. The memory is now an object with one batched `read` and one batched `update`, and a cost judge requires at most one of each per chunk, which a per-token loop cannot meet in any framework; a case with keys wider than the rows a read may carry stops the memory being copied out. The padding rule for the last chunk cannot bind and is not promised. Sizes are small (largest chunk 16) to keep JAX compilation short. [kit: `DeltaMemory`, currently a local helper in the problem's tests]

**`mamba2-ssd-step`** — Implement `ssd_step(h, x, dt_raw, B, C, A_log, D, dt_bias, dt_limit) -> (y, h)` for `H` heads with `h` of shape `(H, P, N)` and `B`, `C` of shape `(G, N)`, each group serving `H / G` consecutive heads: `dt = clip(softplus(dt_raw + dt_bias), dt_limit)`, `h = h * exp(-dt * exp(A_log)) + dt * x B^T`, `y = h C + D x`; and `ssd_matrix(dt, B, C, A_log)` returning per head the lower-triangular `(T, T)` matrix `M[t, s] = (C_t . B_s) * exp(sum of dt_r * A for s < r <= t) * dt_s` (Mamba-2, Dao and Gu, 2024; the one-token path of `transformers` 5.23 `nemotron_h` and `mlx-lm` 0.32 `ssm.py`, whose clamp on `dt` is followed). Goal: the state-space layer of Nemotron 3.5 is attention with a structured mask. Property: `M X + D X` equals the step loop from a zero state. Mutants: output read before the state update; `dt` missing from the input term; positive exponent; groups tiled instead of repeated across heads; decay sum including position `s` itself; clamp omitted (visible when `softplus` falls below `dt_limit[0]`). As built (2026-11-21): the spec's claim that the shipped one-token path clamps `dt` was wrong. `transformers` clamps only in its chunk scan, at `(time_step_min, inf)`; `mlx-lm` clips in both paths, with a limit of `(0, inf)` for the shipped config that never binds. The problem takes `dt_limit` as an argument and applies it in the step (the `mlx-lm` rule), tests both limits binding, and its editorial states the difference.

**`gated-conv-block`** — Implement `conv_block(x, w_in, w_conv, w_out, state) -> (y, state)`: split `x W_in^T` into three equal parts `B, C, h` in that order, convolve `u = B * h` causally and depthwise with kernel `L` over the saved last `L - 1` rows of `u` followed by the new ones (cross-correlation, no activation), and return `(C * conv(u)) W_out^T` with the new last `L - 1` rows of `u` (the `conv` layers of LFM3 1.7B, `Lfm3ShortConv` in `transformers` 5.23, kernel 7). Goal: the cheapest token mixer in a shipped model and its fixed-size state. Property: chunking invariance and causality; numeric. Mutants: symmetric padding that reads the future; state stores `x` instead of `u`; output gate taken from the wrong third (`h` as gate); kernel flipped; state reset on each call. As built (2026-11-21): cross-correlation was confirmed by running `Lfm3ShortConv` with a reversed kernel; the state passed in is not modified.

**`hybrid-decoder-forward`** — Implement `layer_types(n_layers, interval)` (layer `i` is `"full_attention"` when `(i + 1) % interval == 0`, else `"linear_attention"`) and `hybrid_forward(params, cfg, tokens, state) -> (logits, state)` for the kit's hybrid layout: pre-norm blocks whose mixer is `ops.gated_delta_layer` or gated attention (`q_proj` gives `2 * head_dim` per head, query then gate; QK-norm; partial RoPE; `o_proj(attn * sigmoid(gate))`), SwiGLU, final norm, tied head; `state` is a list with `(conv_state, S)` for a recurrent layer and `(k, v)` for an attention layer (the Qwen3.5 text decoder, `transformers` 5.23 and `mlx-lm` 0.32). Goal: one decode loop over two kinds of per-layer state. Property: prefill in any chunking, then one token at a time, equals a single pass; numeric against `hybrid_forward_ref`. Mutants: schedule `i % interval == 0`; a recurrent layer's state not returned (restarts each call); RoPE positions restart at 0 on each call; output gate omitted; query and gate split across the flat projection instead of within each head. [kit: `HybridConfig`, `hybrid_params`, `ops.gated_delta_layer`, `ops.partial_rope`] As built (2026-11-21): staged (schedule, one pass, state and cached decoding); `hybrid_forward` reads `cfg.layers`, so an explicit `layer_types` wins, as in `Qwen3_5TextConfig`; tokens are `(T,)`; embeddings are always tied; the token count comes from the cached `k` of any attention layer, and a model may have none (no position is then needed) or be attention-only. Expected logits come from a tiny `Qwen3_5ForCausalLM` with kit weights (largest difference 3.6e-5 on logits of magnitude 7; tolerance 5e-4).

#### Module `sparse-attention` (order 220)

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `indexer-topk-select` | Scoring the Past with an Indexer | easy 3 | numeric, property | t/j/m | `causal-self-attention` | M6 |
| 20 | `gathered-sparse-attention` | Attention over the Selected Tokens Only | easy 3 | numeric, property, cost | t/j/m | `indexer-topk-select` | M6 |
| 30 | `indexer-cache-decode` | Decoding with an Indexer Cache | medium 2 | numeric, property, cost | t/j/m | `gathered-sparse-attention`, `kv-cache-append` | M6 |
| 40 | `shared-index-layers` | Sharing One Selection Across Layers | easy 2 | numeric, cost | t/j/m | `indexer-topk-select` | M6 |
| 50 | `compressed-kv-stream` | Compressing Keys and Values by Windows | easy 3 | numeric, property | t/j/m | `kv-cache-append`, `stable-softmax` | M6 |

**`indexer-topk-select`** — Implement `index_scores(q_idx, k_idx, w) -> (T, S)` for `q_idx` `(T, H, D)`, one shared key per token `k_idx` `(S, D)`, and raw head weights `w` `(T, H)`: `I[t, s] = sum_h (w[t, h] * H^-0.5) * relu(q_idx[t, h] . k_idx[s] * D^-0.5)`; and `topk_mask(scores, visible, k)` returning the boolean `(T, S)` mask of each row's `min(k, S)` highest scores after setting invisible entries to `-inf`, intersected with `visible`, ties to the lowest index (the lightning indexer of DeepSeek Sparse Attention, DeepSeek-V3.2, 2025, Eq. 1 and 2; scales and masking as in `GlmMoeDsaIndexer`, `transformers` 5.23). Goal: selection is a second, cheaper attention whose scores are never normalized. Property: with `k` at least the number of visible tokens the mask equals `visible`. Mutants: ReLU after the sum over heads; head weights ignored; top-k taken before the future is masked (then intersected, so rows keep fewer than `k`); ReLU also applied to `w`; absolute value in place of ReLU. As built (2026-11-21): the tie rule (ties to the lowest index) is this problem's own, since the shipped `topk` promises no order, and it is tested with equal scores near 1000 and with distinct scores 1e-7 apart, which an index-offset trick gets wrong.

**`gathered-sparse-attention`** — Implement `sparse_attend(q, store, idx)` for queries `(T, H, d)`, a `CountingKV` of `S` keys and values, and `idx` `(T, k)` of selected token indices with `-1` marking an unused slot (the sentinel of `DeepseekV4Indexer`): gather only the selected rows, attend with scale `d^-0.5`, give unused slots zero weight, and return `(T, H, d)`. Goal: the saving of sparse attention comes from never reading the other keys, which a full score matrix with a mask does not achieve. Numeric against masked full attention; property: equal to causal attention when every visible token is selected; cost: `rows_read` equals the number of valid slots. Mutants: `-1` used as an index (reads the last key); unused slots scored 0 instead of `-inf`; every key read and then masked (cost); scale `k^-0.5`; the first query's indices used for every row. `idx` never repeats a token within a row. [kit: `CountingKV`] As built (2026-11-21): the cost is at most the number of valid slots; a row with every slot unused returns zeros; the counters and `with_framework` are off limits, and the store is handed over behind a guard that refuses private attributes, so a forged or reset count is rejected.

**`indexer-cache-decode`** — Implement `dsa_step(dsa, layer, x_new, cache, k) -> (out, cache)` for a chunk of new tokens: compute and append the new tokens' indexer keys and main keys and values, score each new query against all cached indexer keys with causality inside the chunk, select the top `k`, and attend over the gathered rows only; `cache` holds the indexer keys beside the keys and values (the per-layer indexer cache of `GlmMoeDsaIndexer`, `past_key_values.update_indexer`, `transformers` 5.23). Goal: a sparse-attention layer has two caches that must grow together. Property: any chunking equals the full causal pass of the kit oracle; cost: `index_key_rows` equals the number of tokens processed and `rows_read <= k` per query. Mutants: indexer keys recomputed for the whole prefix each call (cost); the new token's own key scored before it is appended (it cannot select itself); no causality within a multi-token chunk; one selection computed for the chunk's last query and used for every row; all keys read (cost). [kit: `TinyDSA`, `CountingKV`] As built (2026-11-21): the kit object comes first, as in `cross-layer-kv-sharing`; the cost rule is per call, at most the sum over new queries of `min(k, visible)` rows and at least the number of distinct earlier-call rows the queries select, which rejects keys stashed outside the cache; the mutant "indexer keys recomputed for the whole prefix" cannot be written through this signature and is replaced by "indexer keys of the new tokens computed twice".

**`shared-index-layers`** — Implement `forward_shared(dsa, h, types, k)` over a stack whose `types[i]` is `"full"` or `"shared"`: a full layer calls `dsa.indexer(i, h)` and attends with the result; a shared layer attends with the indices of the nearest preceding full layer; a shared layer with no full layer before it raises `ValueError` (the cross-layer index reuse of GLM-5.3, `GlmMoeDsaAttention` in `transformers` 5.23; IndexCache, arXiv 2603.12201). Goal: selections agree across neighboring layers, so most indexers can be skipped. Numeric against the oracle; cost: `indexer_calls` equals the number of full layers. Mutants: every layer runs its indexer (cost); shared layers reuse the first full layer's indices instead of the nearest; a shared layer calls the nearest full layer's indexer on its own input (different indices, and extra calls); a shared layer with no full layer before it attends to every token instead of raising. [kit: `TinyDSA`] As built (2026-11-21): the kit object comes first; the `ValueError` must be raised before any indexer or attend call, because the kit's `attend` would otherwise raise on its own and hide a missing check; verified against `GlmMoeDsaModel` (5 indexer calls for 11 layers) and the GLM-5.3 config (78 layers, 21 full).

**`compressed-kv-stream`** — Implement `compress(kv, gate, position_bias, gain, state, rate) -> (entries, state)` for projected rows `kv`, `gate` of shape `(n, d)` arriving in arbitrary pieces: buffer rows until a window of `rate` is complete, emit one entry per complete window as `gain * rms_norm(sum_j softmax_j(gate_j + position_bias_j) * kv_j)` with the softmax over the window per channel, and keep the remainder and the count of entries in `state`; and `visible_entries(t, rate) = (t + 1) // rate`, the number of entries the query at position `t` may attend to (Heavily Compressed Attention of DeepSeek-V4, 2026, as in `DeepseekV4HCACompressor` and `DeepseekV4HCACache`, `transformers` 5.23; the RoPE applied to each entry at position `w * rate` is left out). Goal: a cache that grows by one entry per `rate` tokens and stays causal. Property: every way of cutting the stream gives the same entries; an entry depends only on its own window. Mutants: softmax over channels; position bias omitted; the unfinished window emitted early; the remainder dropped between calls; threshold `t // rate` (a window's last token cannot see it); threshold rounded up (a window visible before it is complete). As built (2026-11-21): the norm's epsilon is 1e-6, inside the root; the state is `(buf_kv, buf_gate, count)`; `visible_entries` must be exact integer arithmetic in constant time, tested up to `t = 2^31 - 1`.

#### Module `block-components` (order 230)

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `partial-rope` | Rotating Part of a Head | easy 3 | numeric, property | t/j/m | `rope` | M6 |
| 20 | `yarn-frequencies` | YaRN: Rescaling Rotary Frequencies | easy 2 | numeric, property | t/j/m | `rope` | M6 |
| 30 | `shared-kv-rotation` | One Tensor as Key and Value | easy 3 | numeric, property | t/j/m | `partial-rope`, `causal-self-attention` | M6 |
| 40 | `sigmoid-router-shared-expert` | Sigmoid Routing with a Shared Expert | easy 3 | numeric | t/j/m | `moe-router-forward` | M6 |
| 50 | `latent-moe` | Experts in a Narrower Space | medium 2 | numeric, cost | t/j/m | `sigmoid-router-shared-expert`, `moe-sparse-dispatch` | M6 |
| 60 | `per-layer-embeddings` | Per-Layer Embeddings | medium 1 | numeric, property | t/j/m | `pre-norm-block` | M6 |
| 70 | `hyper-connections` | Hyper-Connections: Several Residual Streams | medium 1 | numeric, property | t/j/m | `pre-norm-block` | M6 |

**`partial-rope`** — Implement `partial_rope(x, positions, base, factor)`: rotate the first `r = int(d * factor)` channels with frequencies `base^(-2i / r)` in the half-split layout inside those `r` channels and pass the rest (Qwen3.5 with factor 0.25; `transformers` 5.23 `qwen3_5`); and `proportional_rope(x, positions, base, factor)`: `a = int(factor * d // 2)` rotated pairs with frequencies `base^(-2i / d)` and the remaining pairs at frequency zero, half-split over the whole head, pair `i` with `i + d/2` (`rope_type: "proportional"` of Gemma 4, `_compute_proportional_rope_parameters`). Goal: two shipped meanings of `partial_rotary_factor` that give different numbers from the same config values. Property: unrotated channels are returned unchanged; scores depend only on position differences. Mutants: the exponent's denominator swapped between the two functions; the last `r` channels rotated; the proportional form pairing `i` with `i + a` among the first `2a` channels; all channels rotated; adjacent pairs `(0, 1), (2, 3)` rotated instead of the half-split pairs. As built (2026-11-21): the argument is `base`, as in `rope` (configs call it `rope_theta`); `factor` is a multiple of 1/8 so `int(D * factor)` is exact; LFM3 is not an example, because its config says `default` with 0.5 but the installed `Lfm3RotaryEmbedding` ignores the factor on that path.

**`yarn-frequencies`** — Implement `yarn_inv_freq(dim, base, factor, original_max_positions, beta_fast, beta_slow, truncate)` returning `dim / 2` inverse frequencies blended between the original ones and the ones divided by `factor` with the linear ramp of section 9.2, and `yarn_tables(positions, inv_freq, factor) -> (cos, sin)` with both tables multiplied by `0.1 * ln(factor) + 1` (YaRN, Peng et al., 2023; `_compute_yarn_parameters` in `transformers` 5.23; gpt-oss-20b with factor 32 from 4096 and `truncate: false`, DeepSeek-V4 with factor 16 from 65536). Goal: how a model trained at one context length is run at a longer one without retraining. Numeric on the two shipped parameter sets; property: `factor = 1` returns the original frequencies and tables, the fastest frequencies are unchanged, and the slowest are divided by `factor` exactly. Mutants: ramp reversed; `truncate` ignored; `log10` in the table factor; `beta_fast` and `beta_slow` exchanged; table factor applied to `cos` only. As built (2026-11-21): defaults `beta_fast=32, beta_slow=1, truncate=True`; the code's ramp is linear in pair index where the paper's is in rotations, and the problem follows the code; DeepSeek-V4's config class forces the attention factor to 1.0, so the table factor applies to gpt-oss only; tables are compared with a tolerance that grows with the largest position, so float32 and float64 angles both pass; a hidden case makes the upper clamp of the ramp bind (it cannot with the default betas and a base of 32 or more), with a mutant for the missing clamp.

**`shared-kv-rotation`** — Implement `shared_kv_attention(q, kv, pos_q, pos_k, mask, rope_dim, base)` for `q` `(H, T, d)` and one shared tensor `kv` `(S, d)` used as both key and value: rotate the trailing `rope_dim` channels of `q` and `kv` in interleaved pairs, attend with scale `d^-0.5`, then rotate the trailing `rope_dim` channels of each output row by minus its query position (the attention core of DeepSeek-V4, `DeepseekV4Attention` in `transformers` 5.23, without its sinks and compressed entries). Goal: when values carry the rotation, the output must undo it, or the result depends on absolute position. Property: adding a constant to every position leaves the output unchanged; the leading channels equal plain attention. Mutants: no inverse rotation; inverse rotation by plus the position; inverse rotation applied to the leading channels; each value de-rotated by its own key position before the sum (shift-invariant, but the relative rotation is lost); unrotated values used with the inverse rotation still applied. As built (2026-11-21): the argument is `base`; the mutant that de-rotated the leading channels failed every case and was replaced by one that de-rotates by the key positions; a finite additive mask in place of exclusion is a mutant, caught by a case in which a masked row would otherwise win.

**`sigmoid-router-shared-expert`** — Implement `route(x, w_router, bias, k, scale, score_fn) -> (indices, weights)`: `scores = sigmoid(x W^T)` or `sqrt(softplus(x W^T))`; choose the top `k` by `scores + bias`, ties to the lowest index; weights are the chosen experts' unbiased scores divided by `(their sum + 1e-20)` and then multiplied by `scale`; and `moe_shared(x, w_router, bias, experts_w, shared_w, k, scale, score_fn="sigmoid")` adding the always-on shared SwiGLU expert unweighted (DeepSeek-V3, 2024, Eq. 12 to 16; `GlmMoeDsaTopkRouter` and `DeepseekV4TopKRouter` in `transformers` 5.23; GLM-5.3 with scale 2.5, DeepSeek-V4 with `sqrtsoftplus`; one expert group, as in every config read). Goal: experts scored independently, a bias that steers selection without changing the mixture, and a scale that survives the normalization. Numeric. Mutants: softmax scores; weights taken from the biased scores; bias ignored in the selection; scale applied before the normalization (it cancels); shared expert included in the normalized weights; `softplus` without the square root. As built (2026-11-21): `x` is `(N, D)`; an unknown `score_fn` raises `ValueError` in both functions; indices are returned best first, which is this problem's choice (the shipped routers call `topk(sorted=False)`); `sqrtsoftplus` cases keep logits within 5 in magnitude because MLX's `softplus` is inexact for large negative inputs.

**`latent-moe`** — Implement `latent_moe(x, w_router, w_down, w_up, norm_gain, experts, shared_w, k)`: route on the full-width `x` with a sigmoid router the statement defines (no bias, scale 1), project `x` to the latent width once, call each chosen expert on its rows there, form the weighted sum, apply the RMS norm at latent width, project up, and add the full-width shared expert; and `expert_params(hidden, latent, inter, n_experts)` returning the parameter counts of the routed experts with and without the latent projection (the latent experts of Kimi K3.1, `routed_expert_hidden_size` 3584 of 7168, as in `mlx-lm` 0.32 `kimi_k3.py`; Nemotron-H has the same form without the norm). Goal: hundreds of experts are affordable when each works at half width. Numeric; cost: `expert_rows == k * n_tokens`. Mutants: norm applied to each expert's output before the weighted sum; norm omitted; dense evaluation (cost); the two projections counted once per expert in `expert_params`; routing weights not applied. [kit: `CountingExperts`] As built (2026-11-21): the norm epsilon is 1e-5, the shipped `rms_norm_eps`; `k` is at least 2, because `mlx-lm` skips the weight normalization at `k = 1`; calling an expert no token chose is forbidden and counted; the unnormalized-weights mutant was dropped because the norm nearly cancels it; `CountingExperts` already existed in the kit.

**`per-layer-embeddings`** — Implement `ple_inputs(tokens, h0, table, w_model, norm_gain, n_layers)`: the token part is `table[tokens] * sqrt(d_ple)` reshaped from `(T, L * d_ple)` to `(T, L, d_ple)`; the context part is `(h0 W_model^T) * hidden^-0.5` reshaped the same way and RMS-normalized over `d_ple`; return their sum times `2^-0.5`; and `ple_inject(h, ple_i, w_gate, w_proj, post_gain)` returning `h + rms_norm(proj(gelu_tanh(gate(h)) * ple_i))`, applied in each layer after the MLP residual (the per-layer embeddings of Gemma 4 E2B and E4B, `Gemma4TextModel` and `Gemma4TextDecoderLayer` in `transformers` 5.23; `hidden_size_per_layer_input` 256). Goal: where a model's "effective" parameter count hides a second, larger embedding table that costs a lookup instead of a matrix product. Numeric; property: `ple_inputs` for layer `i` reads only columns `i * d_ple` to `(i + 1) * d_ple` of the table. Mutants: packed row reshaped as `(d_ple, L)`; scale `sqrt(hidden)`; context part omitted; layer 0's slice used in every layer; the `2^-0.5` omitted; the activation on the gate omitted. As built (2026-11-21): a third function, `ple_layer(params, x, ple, i, mask)`, makes the injection position and the layer slice bind; its attention is the kit's single-head stand-in, and the statement says so. Verified against `Gemma4TextModel` and `Gemma4TextDecoderLayer` (largest difference 1.3e-6) and the E2B (35 layers) and E4B (42) configs, both with `enable_moe_block: false`.

**`hyper-connections`** — Implement `hc_weights(streams, fn, base, scale, iters, eps) -> (collapsed, post, comb)` for `streams` `(T, N, D)`: flatten to `(T, N * D)`, apply the unweighted RMS norm, one linear map to `N + N + N * N` values, `pre = sigmoid(pre_w * scale[0] + base_pre) + eps`, `post = 2 * sigmoid(post_w * scale[1] + base_post)`, `comb = softmax(comb_w * scale[2] + base_comb, axis=-1) + eps`, then normalize `comb` by column sums and repeat `iters - 1` times row sums then column sums, each divided by `(sum + eps)`; `collapsed = sum_n pre_n * streams_n`; and `hc_merge(streams, y, post, comb)` returning `post[..., None] * y[:, None, :] + comb^T @ streams` (manifold-constrained hyper-connections of DeepSeek-V4, `DeepseekV4HyperConnection` and `DeepseekV4DecoderLayer` in `transformers` 5.23; 4 streams, 20 iterations). Goal: a residual stream replaced by several, mixed by a matrix kept doubly stochastic so depth does not amplify it. Numeric with `iters` of 1, 2, and 20; property: at 20 iterations rows and columns of `comb` sum to 1 within 1e-4. Mutants: `comb` applied untransposed; no Sinkhorn normalization; `post` without the factor 2; streams averaged instead of weighted by `pre`; RMS norm per stream instead of over the flattened vector. Not a mutant: starting the normalization with rows, which converges to the same matrix at 20 iterations (difference 2e-14 measured). As built (2026-11-21): the property "at 20 iterations rows and columns of `comb` sum to 1 within 1e-4" is false for sharp matrices (row error up to 8e-2). The problem states: columns within 1e-5 at every `iters`, rows within 1e-4 at 20, both when the softmax inputs span at most 2. Starting with rows is a mutant, because it binds at 1 and 2 iterations. Bit-exact against `DeepseekV4HyperConnection`.

#### Module `drafts-and-diffusion` (order 240)

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `mtp-chain-draft` | Drafting with a Chained Multi-Token-Prediction Module | medium 3 | numeric | t/j/m | `greedy-speculative-verify`, `pre-norm-block` | M6 |
| 20 | `entropy-bound-denoise-step` | One Denoising Step with Entropy-Bounded Acceptance | medium 1 | numeric, property | t/j/m | `categorical-sample` | M6 |
| 30 | `block-diffusion-decode` | Block Text Diffusion over a Cached Prefix | medium 3 | numeric, cost | t/j/m | `entropy-bound-denoise-step`, `kv-cache-append` | M6 |

**`mtp-chain-draft`** — Implement `mtp_chain_draft(params, cfg, tokens, next_token, hidden, n_draft) -> (draft_tokens, draft_logits)` for the `T` tokens the main model has processed, the token it then produced, and its final-normed hidden states `(T, D)`: stage `j` takes the token ids shifted by `j + 1` (ending with the latest drafted token) and the previous stage's output (the main hidden states for `j = 0`), computes `eh_proj(concat(enorm(E[ids]), hnorm(h_prev)))`, runs its own decoder block at positions `j + 1 .. j + T` composed from kit `ops`, applies its post norm, and takes the argmax of the shared head at the last row (the MTP module of DeepSeek-V3, 2024, Eq. 21 to 23, as implemented by `MtpLayer` and `MtpModel` in `transformers` 5.23; Qwen3.5 checkpoints ship one such layer as `mtp.*` tensors, and the GLM-5.3 and DeepSeek-V4.1 configs declare 1 and 3; the embedding is concatenated first as in `transformers`, where the paper writes the hidden state first). Goal: the draft model that ships inside the checkpoint, and how it differs from the independent heads of `mtp-head-draft`. Numeric against `mtp_draft_ref` for one, two, and three stages. Mutants: concatenation order swapped; the embedding of the current token instead of the next; every stage reads the main model's hidden states; the drafted token not appended for the next stage; post norm omitted; positions not shifted per stage. [kit: `mtp_params`] As built (2026-11-21): renamed `mtp_chain_draft`, so it does not share a name with the function of `mtp-head-draft`, with `params, cfg` first; `1 <= T <= 8`. The mutant "positions not shifted per stage" cannot fail, because rotary scores are shift-invariant when there is no cache (measured 2.5e-6), and is replaced by a missing causal mask. The block is the kit's decoder block, not a real Qwen or GLM block.

**`entropy-bound-denoise-step`** — Implement `step_temperature(n, n_max, t_min, t_max) = t_min + (t_max - t_min) * n / n_max` for `n` steps remaining, and `denoise_step(current, logits, temperature, entropy_bound, u_sample, u_noise) -> (canvas, accepted, argmax_canvas)` for one canvas of `L` positions: divide the logits by the temperature, sample one token per position by inverse CDF from `u_sample`, compute each position's entropy in nats, sort ascending (ties to the lowest index), accept positions while the sum of the entropies before a position in that order is at most `entropy_bound`, write the sampled tokens at accepted positions and `floor(u_noise * V)` at every other position (the entropy-bound sampler of DiffusionGemma, `EntropyBoundSampler` and `_denoising_step` in `transformers` 5.23; the acceptance rule of arXiv 2505.24857). Goal: commit several tokens per pass only when they are nearly independent, and note that this sampler recomputes acceptance from scratch each step. Property: the lowest-entropy position is always accepted; a larger bound never accepts fewer positions. Mutants: the position's own entropy included in the sum; each entropy compared with the bound separately; descending order; rejected positions keep their current token; entropy of the logits before the temperature; temperature schedule reversed. As built (2026-11-21): ties in the entropy sort go to the lowest index, tested on 256 positions with 100 identical rows astride the cut, because PyTorch's default sort is stable only up to 16 elements and the shipped sampler uses it; distinct entropies are at least 1e-5 apart; `<=` versus `<` on the bound is not tested and the statement says so. Exact against `_denoising_step` with `EntropyBoundSampler` on 9 runs.

**`block-diffusion-decode`** — Implement `diffusion_generate(model, prompt, cfg, uniforms) -> tokens`: encode the prompt into the cache once; for each canvas, start from random tokens, run at most `cfg.max_steps` denoising steps with `ops.eb_accept`, passing each step's temperature-scaled logits to the next call as self-conditioning; stop a canvas when its argmax canvas equaled each of the previous `cfg.stability` argmax canvases and the mean entropy is below `cfg.confidence`; append the argmax canvas of the last step run, replace everything after the first EOS in it by `pad_id` and stop, or encode the canvas into the cache and continue up to `cfg.max_canvases` (the `generate` loop of DiffusionGemma, `transformers` 5.23, single sequence). Goal: a text-diffusion model is autoregressive over blocks, and its prefix cache works like any other. Numeric on exact tokens; cost: `encoder_tokens == P + (canvases - 1) * L` and `denoise_calls` equal to the oracle's. Mutants: the whole sequence encoded again for each canvas (cost); the sampled canvas appended instead of the argmax canvas; self-conditioning logits not carried between steps; stop on stability alone; the noisy canvas encoded into the cache; no early stop (cost). [kit: `BlockDenoiser`, `ops.eb_accept`] As built (2026-11-21): staged (one canvas, cached prefix, cost); `cfg` is a dict, `uniforms` is `(u_init, u_sample, u_noise)`, and the result includes the prompt; from the source, `stability = 0` is always stable and `encoder_calls == canvases`; the special-token ids are read from the model and vary between cases; the model is handed over behind a whitelist guard and rejects a cache it did not produce. The loop agrees with `DiffusionGemmaGenerationMixin.generate` on 26 scenarios over the toy `BlockDenoiser`; the denoiser itself is not a shipped model. The library's docstring says stability compares the accepted canvas; its code compares argmax canvases, and the problem follows the code.

#### Module `real-hybrid-model` (order 250)

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `layer-plan-from-config` | A Layer Plan from `config.json` | medium 1 | numeric | python | - | M6 |
| 20 | `real-gdn-block` | One Recurrent Block of a Real Model | medium 3 | numeric | t/j/m | `gated-delta-layer`, `real-model-block` | M6 |
| 30 | `real-hybrid-logits` | Matching a Hybrid Model's Logits | hard 1 | numeric | t/j/m | `real-gdn-block`, `hybrid-decoder-forward`, `partial-rope`, `real-model-logits` | M6 |

**`layer-plan-from-config`** — Implement `layer_plan(config) -> list[dict]` with one entry per layer holding `mixer`, `mlp`, `indexer`, `rope`, and `kv` for three families, from dicts with the keys of the real files: Qwen3.5 (`layer_types` if present, else `(i + 1) % full_attention_interval`; `rope` with `rotary_dim = int(head_dim * partial_rotary_factor)`); GLM-MoE-DSA (`mlp_layer_types` if present, else `min(first_k_dense_replace, L)` dense layers then sparse; `indexer_types` if present, else `index_topk_pattern` with `F` and `S`, else full when `max(i - index_skip_topk_offset + 1, 0) % max(index_topk_freq, 1) == 0`); Gemma 4 (`layer_types`, the per-type `rope_parameters` and head dimension, and `kv` as `"reuse"` for the last `num_kv_shared_layers` layers, `"store"` for the last earlier layer of each type, else `None`) (`Qwen3_5TextConfig`, `GlmMoeDsaConfig`, `Gemma4TextConfig.kv_sharing_modes` in `transformers` 5.23; the configs of `Qwen/Qwen3.5-0.8B`, `zai-org/GLM-5.3`, `google/gemma-4-E2B-it`). Goal: a modern model is a per-layer plan, and the defaults that expand a short config into it are where ports go wrong. Numeric on reduced copies of the three real configs and on hidden synthetic ones; no weights. Mutants: schedule `i % interval`; an explicit list ignored in favor of the rule; the index rule without the clamp at zero (layer 0 becomes shared); offset applied as `i - offset`; `"store"` given to the last unshared layer only, not to one per type; the sliding layers' RoPE parameters used for full layers. As built (2026-11-21): the `rope` value is `None` or `{type, theta, head_dim, rotary_dim}`, and `indexer` and `kv` are `None` where a family has none. The decoder's settings are read from `text_config` when the file has one; an unknown or missing `model_type` raises `ValueError`; Qwen's missing `partial_rotary_factor` is 0.25 and Gemma's is 1.0; `global_head_dim` defaults to 512. Expected plans come from the three configuration classes and the models' rotary embeddings through `fixtures/make.py`, not from the reference. `int(head_dim * f)` is constrained to be even, because the library's rotary embedding rounds an odd value up (found by the reviewer's fuzz of 1,400 configs). The spec's `min(first_k_dense_replace, L)` is unobservable and is not a mutant. 16 mutants; the three rules that only hidden cases distinguished (Gemma's missing factor, the default global head dimension, `store` for a type no shared layer reuses) now have public cases.

**`real-gdn-block`** — Implement `qwen_gdn_block(x, weights, layer, state) -> (x, state)` for a linear-attention layer of `Qwen/Qwen3.5-0.8B` from the raw tensors: RMSNorm with eps 1e-6 and the stored weight used as `1 + weight`; the delta layer of `gated-delta-layer` with 16 key and 16 value heads of 128 and the `(6144, 1, 4)` convolution; its gated norm, whose weight is used as stored; residual; second norm; SwiGLU of width 3584; residual. Numeric against fixture hidden states for two linear layers, one early and one in the middle, produced by `transformers` on prompts of at most 16 token ids (`rtol` 1e-3, to be confirmed by measurement); synthetic public cases run without the asset. Goal: the conventions of one real checkpoint that no small synthetic model reveals. Mutants: block norm weights used without the `1 +` (real-weight cases only); the `1 +` also applied to the gated norm; `dt_bias` omitted; convolution kernel flipped; BF16 weights not converted to float32. [kit: asset `qwen3.5-0.8b`] As built (2026-11-21): returns `(x, (conv_state, S))`. Tolerance `rtol` 1e-3 with `atol` 1e-5 (hidden entries are 0.02 to 0.04, so the proposed `atol` of 1e-3 would pass anything); fixtures are bit-identical to a fresh `transformers` 5.23.0 run and the reference agrees within 3e-7. The mutant "BF16 not converted" cannot be written (the kit's `Weights` always decodes) and is replaced by `computes_in_bfloat16`; `norm_eps_default`, `conv_state_restarts`, `ignores_state`, and `state_updated_in_place` are added. The `1 +` mutant is caught by synthetic cases too; the real-weights-only mutant is `layer_prefix_without_dot` (layer 1 against layers 10 to 19). Added after review: a case that the arguments keep their values, calls of one row (a 4 + 1 + 1 split), and the per-framework backends of `ops.gated_delta_layer` in `forbidden` (a reviewer's solution passed by calling one). 14 mutants; 13 cases, 4 on real weights.

**`real-hybrid-logits`** — Implement `qwen_logits(weights, cfg, tokens)`: the 24-layer text decoder of `Qwen/Qwen3.5-0.8B` from a name-to-array dict. The list a solution must get right: embeddings unscaled; 18 delta blocks and 6 attention blocks by `layer_types`; attention with 8 heads over 2 KV heads of 256, `q_proj` split per head into query and gate, RMS QK-norm with `1 + weight`, RoPE with theta 1e7 on the first 64 channels in the half-split layout inside them, scale `256^-0.5`, `o_proj(attn * sigmoid(gate))`; every block norm and the final norm with `1 + weight`; the head is the embedding matrix. Judged against fixture logits from `transformers` on prompts of at most 16 token ids (`rtol` 1e-3, `atol` 1e-3 expected, to be measured) and the fixture argmax tokens exactly. Goal: porting a hybrid checkpoint with no reference code open. Mutants: output gate omitted; RoPE over the whole head; final norm without the `1 +`; QK-norm weights used without the `1 +`; the query and gate halves of `q_proj` exchanged. [kit: asset `qwen3.5-0.8b`, `RealHybridLM`] As built (2026-11-21): `tokens` is one sequence `(T,)`. Tolerance as measured, `rtol` 1e-3 and `atol` 1e-3, with the argmax compared where the fixture's top-two margin is at least 0.01 (every margin is at least 0.044); the reference agrees with `transformers` within 6.1e-5. Two real-weight cases (one public, one hidden) keep JAX under the 3 s warning. Added after review: models generated and judged at run time against the kit's numpy forward, because a solution that replayed the fixture files was accepted; a synthetic model with `norm_eps` 0.01 so that a default epsilon in the block norms or in the QK norms fails widely (`norm_eps_default`, `qk_norm_eps_default`); a one-token case. Removed after review: a hidden case that injected an `lm_head.weight` although the statement says there is none, which rejected an honest port. The real-weights-only mutant is `layers_in_name_order`. The editorial records that `mlx-lm` 0.32.2 disagrees with `transformers` on this checkpoint because it adds the norms' 1.0 in bfloat16 (0.26 to 0.70 in the logits as loaded, depending on the prompts). 15 mutants; 9 cases.

### 9.6 Author batches

Every proposed problem is in exactly one batch. Kit work under "needs" lands first.

| batch | theme | problems | needs |
|---|---|---|---|
| A | Positions and experts | `partial-rope`, `yarn-frequencies`, `shared-kv-rotation`, `sigmoid-router-shared-expert`, `latent-moe` | |
| B | The delta rule and its neighbors | `gated-delta-step`, `gated-delta-layer`, `gated-delta-chunked`, `mamba2-ssd-step`, `gated-conv-block` | `gdn_params`, `ops.l2norm`, `ops.causal_conv1d` |
| C | Indexed attention | `indexer-topk-select`, `gathered-sparse-attention`, `indexer-cache-decode`, `shared-index-layers`, `compressed-kv-stream` | `CountingKV`, `TinyDSA` |
| D | Streams and steps | `per-layer-embeddings`, `hyper-connections`, `mtp-chain-draft`, `entropy-bound-denoise-step` | `mtp_params` |
| E | Whole loops (capstone-like) | `hybrid-decoder-forward`, `block-diffusion-decode` | `HybridConfig`, `hybrid_params`, `ops.gated_delta_layer`, `ops.partial_rope`, `BlockDenoiser`, `ops.eb_accept` |
| F | The real hybrid model | `layer-plan-from-config`, `real-gdn-block`, `real-hybrid-logits` | asset `qwen3.5-0.8b`, `RealHybridLM`; the owner's decision on the download |

Suggested order: A needs no kit and can be the review checkpoint for statement style;
then one kit pass for B, C, D, and E; then B, C, D, E in that order (E's kit oracles
reuse what B and D exercised); F last, after the download decision. Batch F has three
problems because two of them are real-weight problems of capstone weight. In batch B
write `gated-delta-step` first; in batch C write `indexer-topk-select` first.

### 9.7 Deferred components

Not specified. Each needs the stated work before it can be.

| component | where it appears | status and reason |
|---|---|---|
| Compressed Sparse Attention (CSA) | DeepSeek-V4, V4.1 | Mechanics VERIFIED in `transformers` 5.23 `deepseek_v4` (two overlapping series per token, an indexer over compressed keys, a `-1` sentinel). Deferred for size: it is `compressed-kv-stream` plus `indexer-cache-decode` plus an overlap state, too much for one problem before those two exist. |
| Attention residual blocks | Kimi K3.1 (`attn_res_block_size` 12), Inkling 2 | Read in `mlx-lm` 0.32 `kimi_k3.py` only (a softmax over stored block-boundary streams and the running partial sum, logits from an RMS-normalized projection). Moonshot's own code and any paper were not read, and the file implements `kimi_k3`, not `kimi_k31`. One more source is needed. |
| Hashed n-gram embeddings ("Engram") | DeepSeek-V4.1 (`engram_*` keys) | UNVERIFIED. Config keys only; `mlx-lm` `deepseek_v41.py` was not read; `transformers` 5.23 has no `deepseek_v41`. The survey's guess that V4's `num_hash_layers` is the same thing is wrong: it selects hash-routed expert layers. |
| Markov or RNN draft heads ("DSpark") | DeepSeek-V4 and V4.1, Inkling 2, Mistral Large 4 | UNVERIFIED. `transformers` 5.23 ignores `mtp_dspark.*` weights; no implementation or paper read. |
| Block-diffusion draft model (DFlash) | `deepseek-ai/dflash_*` | Candidate generator VERIFIED (`DFlashTokenCandidateGenerator`: one pass over an anchor token and mask tokens, reading target hidden states). The draft model's own layers were not read. |
| Hash-routed expert layers | DeepSeek-V4 (`num_hash_layers` 3) | VERIFIED (`DeepseekV4HashRouter`: experts chosen by a frozen token-id table, weights from the learned scores). Small; a candidate extension of `sigmoid-router-shared-expert`. |
| Parallel dense and expert branches | Gemma 4 26B-A4B, DiffusionGemma | VERIFIED (`Gemma4TextDecoderLayer` with `enable_moe_block`; softmax router with per-expert scale). Left out to keep the count near 20. |
| Per-layer-type RoPE as a tensor problem | Gemma 4, LFM3, MiMo | Covered instead by `partial-rope` (the arithmetic) and `layer-plan-from-config` (which parameters a layer uses). A separate tensor problem would repeat `sliding-window-attention`. |
| Kimi delta attention variants | Kimi K3.1 (`use_full_rank_gate`, `gate_lower_bound`) | Partly read (`compute_lower_bound_g` in `mlx-lm`); not specified. |
| Other models' specifics | Inkling 2 (relative positions, QK clipping), Mistral Large 4 (layers without RoPE, KV source layers), MiMo (sink bias, value scale), Qwen3.8-Flash-Next (`qwen4_exp`), Olala | UNVERIFIED. Modeling files exist in the installed `transformers` and were not read. |
| M-RoPE, vision towers, audio | Qwen3.5, Gemma 4 | Out of scope (vision-language). |

### 9.8 Corrections to the manager's outline

- **Per-layer-type RoPE** is not a separate tensor problem (see 9.7). **A dense-first-layers
  schedule** is one rule inside `layer-plan-from-config`, not a problem.
- **"Sharing an indexer across layers"** shares the selected indices, not the indexer:
  a shared layer has no indexer weights at all (`GlmMoeDsaAttention`).
- **Multi-token prediction** as shipped is a chain of full decoder blocks, each reading
  the previous one (DeepSeek-V3, Eq. 21 to 23), not parallel heads; `transformers`
  implements it generically and loads the `mtp.*` weights of the checkpoints.
- **Text diffusion** as shipped (DiffusionGemma) is uniform-noise diffusion with
  entropy-bounded acceptance and renoising, not the masked diffusion of
  `diffusion-unmask-step`; it is split into a step and a loop.
- **Held-back items now sourced and included:** hyper-connections and heavily compressed
  attention (both read in `transformers` 5.23 `deepseek_v4`). **Added from the sources:**
  `shared-kv-rotation`.
- **Survey slips found:** Gemma 4 E2B is 4 sliding layers to 1 full with 20 KV-shared
  layers; GLM-5.3 runs 3 full indexers before the 3-shared-1-full pattern; DeepSeek-V4's
  hash layers are an expert router.

### 9.9 Decisions (open when proposed; resolved 2026-11-21)

The owner approved the proposal with these defaults: all 24 problems (1); the download
(2); the names as proposed, in a new area file (3); implementation over paper (4); the
kit as listed (5). Item 6 was settled by measurement: a compiled per-token loop met the
speed limit, so `gated-delta-chunked` has a cost judge on a counting memory instead.
Item 7 was fixed in `f6acbff`: the wrong claim about `mtp-head-draft` was in this
document's spec, not in the problem. Item 8: `M6` stands.

1. **Count.** 24 problems against "about 20". The four to cut first, in order:
   `gated-conv-block`, `latent-moe`, `shared-kv-rotation`, `shared-index-layers`.
2. **The download.** `Qwen/Qwen3.5-0.8B`, 1.75 GB, Apache-2.0, for two problems; about
   15% of the file (vision tower, MTP module) is unused. Alternative: no real-weight
   problems, and batch F shrinks to `layer-plan-from-config`.
3. **Names.** Area id and title, the five module ids and titles, and whether this is a
   new area file `areas/model-architectures.toml` or more modules of the existing area.
4. **Implementation over paper.** Three specs follow the shipped code where the paper
   differs (MTP concatenation order, the sampler that renoises accepted positions, the
   `dt` clamp). This is the catalog's rule; confirm it holds here.
5. **Kit size.** Seven kit additions, of which `TinyDSA`, `BlockDenoiser`, and
   `RealHybridLM` are the large ones.
6. **A speed judge** in `gated-delta-chunked`. The margin must be measured in all three
   frameworks; if it is not at least 5x, the problem keeps only the numeric and property
   judges and loses the mutant that loops per token.
7. **Existing problems this research touches** (not changed here): `mtp-head-draft` says
   parallel heads are what "shipped with DeepSeek-V3", which the report contradicts;
   `cross-layer-kv-sharing` has upper layers reuse the layer just below, where Gemma 4
   reuses the last unshared layer of the same type; `hybrid-state-checkpoints` names
   Qwen3-Next where Qwen3.5 now ships the same layers.
8. **Milestone label.** `M6` is a placeholder.

## 10. Tensor Foundations (2026-11-27)

A module added in front of the catalog after the owner's first five solves: the tensor
mechanics that the other modules assume. It is the first module of the "Inference and
serving" area (track `tensors`, order 5, before `foundations`). The list comes from a
count of the operations in every reference solution (`reshape` 423, `None` indexing 373,
`where` about 340, `arange` about 270, `transpose` 138, `argsort` about 110, `einsum`
86, `gather` and `take_along_axis` about 80, `repeat` and `repeat_interleave` about 75,
`cumsum` about 60) and from the owner's own wrong submissions (a bare `squeeze()` that
dropped a batch axis of size 1; `amax` before `abs`).

Rules for this module, in addition to section 1:

- **Drills.** Each problem is two to four small functions around one mechanic, all
  listed in `entry` and graded together. Each function has at least one public case.
- **No built-in is forbidden** unless the spec says so: the goal is knowing the
  framework's tools, so the statement names them. Any correct solution passes.
- **Shapes that expose the mistake.** Every function has cases where an axis has size 1
  and cases where two axes have equal size (the shapes on which a wrong axis or a wrong
  pairing still returns the right shape), next to cases where all sizes differ.
- **Three frameworks.** The editorial ends with a table of the operation in PyTorch,
  JAX, and MLX, checked by running each, and says where they differ.
- **Integer results** are compared exactly; float results are float32.
- No kit additions are needed.

### Track `tensors` Tensor Foundations (order 5)

| order | id | title | diff | judges | fw | prereq | ms |
|---|---|---|---|---|---|---|---|
| 10 | `broadcast-align` | Broadcasting on Purpose | easy 1 | numeric | t/j/m | - | M7 |
| 20 | `size-one-axes` | Axes of Size One | easy 1 | numeric | t/j/m | `broadcast-align` | M7 |
| 30 | `reshape-and-memory-order` | Reshape, Transpose, and Memory Order | easy 2 | numeric, property | t/j/m | - | M7 |
| 40 | `paired-and-outer-indexing` | Paired and Outer Indexing | easy 2 | numeric | t/j/m | - | M7 |
| 50 | `gather-along-axis` | Gathering Along an Axis | easy 2 | numeric | t/j/m | `paired-and-outer-indexing` | M7 |
| 60 | `scatter-writes` | Scatter and Indexed Writes | easy 3 | numeric | t/j/m | `gather-along-axis` | M7 |
| 70 | `masks-and-where` | Masks and `where` | easy 2 | numeric | t/j/m | `broadcast-align` | M7 |
| 80 | `arange-comparisons` | Masks from `arange` | easy 2 | numeric | t/j/m | `broadcast-align` | M7 |
| 90 | `sort-and-unsort` | Sorting and Unsorting | easy 3 | numeric, property | t/j/m | `gather-along-axis`, `scatter-writes` | M7 |
| 100 | `cumsum-tricks` | Cumulative Sums as Control Flow | easy 3 | numeric | t/j/m | `masks-and-where` | M7 |
| 110 | `einsum-contractions` | Contractions with `einsum` | easy 2 | numeric | t/j/m | - | M7 |
| 120 | `expand-and-repeat` | Expand, Repeat, and Interleave | easy 2 | numeric | t/j/m | `broadcast-align` | M7 |
| 130 | `join-pad-split` | Joining, Padding, and Shifting | easy 3 | numeric | t/j/m | - | M7 |
| 140 | `dtypes-and-promotion` | Dtypes and Promotion | easy 3 | numeric | t/j/m | - | M7 |
| 150 | `loops-to-tensors` | From Loops to Tensors | medium 1 | numeric, speed | t/j/m | `scatter-writes`, `cumsum-tricks` | M7 |

**`broadcast-align`** — Implement `relative_positions(T)` returning the integer `(T, T)` table `out[i, j] = i - j`; `scale_rows(x, s)` multiplying each row `b` of `x` of shape `(B, T, D)` by `s[b]` with `s` of shape `(B,)`; and `pairwise_sq_dist(a, b)` mapping `(N, D)` and `(M, D)` to the `(N, M)` table of squared distances. Goal: shapes align from the right, and `None` puts an axis where you need it. Numeric. Mutants: `j - i`; `x * s` with no added axes (right only by accident when `B == D`); the scale on the `T` axis; elementwise `a - b` that returns `(N,)` distances when `N == M`.

**`size-one-axes`** — Implement `normalize_over(x, dim)` dividing `x` (any rank from 1 to 4, `dim` possibly negative) by its sum over `dim`; `last_step(x)` mapping `(B, T, D)` to `(B, D)`; and `column(x, j)` mapping `(B, V)` to `(B, 1)`. Goal: `keepdim`, and an index or `squeeze(dim)` in place of a bare `squeeze()`, which also drops a batch or feature axis of size 1. Numeric, with `B = 1`, `T = 1`, `D = 1`, and all three at once. Mutants: no `keepdim` (wrong values on a square input); bare `squeeze()`; `x[:, j]` returning `(B,)`; a positive-only `dim`.

**`reshape-and-memory-order`** — Implement `group_columns(w, g)` mapping `(R, C)` to `(R, C // g, g)`; `interleave(a, b)` mapping two `(..., n)` tensors to `(..., 2n)` with `out[..., 2i] = a[..., i]` and `out[..., 2i + 1] = b[..., i]`; and `patchify(x, p)` mapping `(B, H, W)` to `(B, (H // p) * (W // p), p * p)`, patches in row-major order and each patch flattened row-major. Goal: reshape reads elements in row-major order and never moves them, so a regrouping across axes needs a transpose between two reshapes. Numeric, plus the property that every function is a permutation of its input. Mutants: `interleave` by concatenation; `patchify` by reshape alone; patches in column-major order; groups taken along `R`.

**`paired-and-outer-indexing`** — Implement `pick_pairs(x, rows, cols)` returning the `(N,)` values `x[rows[n], cols[n]]`; `pick_block(x, rows, cols)` returning the `(R, C)` block `x[rows[r], cols[c]]`; and `pick_per_row(x, cols)` mapping `x` of shape `(B, T, D)` and `cols` of shape `(B, K)` to `(B, K, D)` with `out[b, k] = x[b, cols[b, k]]`. Goal: index arrays are broadcast against each other and paired, so a cross product needs `rows[:, None]`. Numeric. Mutants: the block by paired indexing (the diagonal when `R == C`); pairs by chained indexing `x[rows][:, cols]`; `x[:, cols]` for the per-row pick; `rows` and `cols` swapped.

**`gather-along-axis`** — Implement `token_scores(logits, ids)` mapping `(B, T, V)` and `(B, T)` to the `(B, T)` scores `logits[b, t, ids[b, t]]`; `take_columns(values, idx)` mapping `(B, V)` and `(B, K)` to `(B, K)`; and `reorder_steps(x, perm)` mapping `(B, T, D)` and `(B, T)` to `out[b, t] = x[b, perm[b, t]]`. Goal: `gather` and `take_along_axis` need an index of the same rank as the source, so the index gains an axis and is expanded. Numeric. Mutants: the gather on the `T` axis; an index used as `logits[:, :, ids]`; `reorder_steps` with an unexpanded index (only feature 0 is right); `perm` applied to the batch axis.

**`scatter-writes`** — Implement `one_hot_rows(ids, V)` mapping `(B,)` to float32 `(B, V)`; `token_counts(ids, V)` mapping `(B, T)` to the integer `(B, V)` count of each token in each row; and `write_at(x, idx, values)` returning a copy of `x` of shape `(B, V)` with `out[b, idx[b, k]] = values[b, k]` (`idx` of shape `(B, K)`, no repeats within a row). Goal: the write that mirrors gather, its three spellings (`scatter`, `.at[]`, `put_along_axis` or indexed assignment), and that repeated indices need the adding form. Numeric, with repeated tokens. Forbid the frameworks' `one_hot`. Mutants: counts by a setting scatter (a repeat counts once); counts over the flattened batch; the scatter along the batch axis; `idx` and `values` swapped in meaning (a write of `idx` values).

**`masks-and-where`** — Implement `fill_masked(scores, keep)` returning `scores` with `-inf` where `keep` is false; `masked_mean(x, mask)` mapping `(B, T, D)` and a boolean `(B, T)` to the `(B, D)` mean over kept steps, zero for a row that keeps none; and `masked_max(x, mask)`, `-inf` for such a row. `x` may hold `inf`, `-inf`, or NaN at dropped steps. Goal: select with `where` instead of multiplying by the mask (`0 * inf` is NaN), and read through a mask without boolean indexing, which MLX does not have. Numeric. Mutants: multiplication by the mask; the mean divided by `T`; the max over `x * mask` (zero beats a negative row); `-1e9` for `-inf`; NaN for an empty row.

**`arange-comparisons`** — Implement `length_mask(lengths, T, side)` returning the boolean `(B, T)` mask of real tokens for right or left padding; `band_mask(T, w)` returning the `(T, T)` mask `0 <= i - j < w`; and `same_segment(seg)` mapping `(T,)` segment ids to the `(T, T)` mask `seg[i] == seg[j]`. Goal: a mask is a comparison between broadcast index vectors. Numeric. Mutants: `<=` for `<` in the length mask; the sides swapped; the band without its lower bound; the band transposed; the window one too wide.

**`sort-and-unsort`** — Implement `rank_descending(x)` mapping `(B, V)` to the indices that sort each row from largest to smallest, equal values by lowest index first; `inverse_permutation(perm)` mapping `(B, V)` to the permutation that undoes each row; and `keep_to_original_order(keep_sorted, order)` mapping a boolean `(B, V)` decision made in sorted order back to token order. Goal: the sort, decide, unsort pattern of every sorted sampling filter, with a stable tie rule in each framework (guide section 6, "Sorting and ties"). Numeric with ties; the property that sorting then unsorting is the identity. Mutants: ties by highest index (an ascending sort reversed); the inverse taken as `argsort` of the values; unsorting by indexing with `order` again (right only for an involution); an unstable sort on a tie-heavy row.

**`cumsum-tricks`** — Implement `exclusive_cumsum(x)` over the last axis (`out[..., i]` is the sum of the entries before `i`); `first_true(mask)` mapping a boolean `(B, T)` to the index of the first true entry of each row, `T` for a row with none; and `through_first_eos(ids, eos)` returning the boolean `(B, T)` mask that is true up to and including the first `eos` of each row, all true for a row with none. Goal: a running sum turns "has it happened yet" into a tensor operation. Numeric. Mutants: the inclusive sum; `argmax` alone (0 for a row with none); the mask that drops the `eos` itself; `ids != eos` (true again after a second `eos`); the sum over the batch axis.

**`einsum-contractions`** — Implement `attention_scores(q, k)` mapping `(B, H, T, D)` and `(B, H, S, D)` to `(B, H, T, S)`; `bilinear(x, w, y)` mapping `(B, D)`, `(D, E)`, and `(B, E)` to the `(B,)` values `x[b] @ w @ y[b]`; and `per_head_project(x, w)` mapping `(B, H, T, D)` and `(H, D, E)` to `(B, H, T, E)`, head `h` using `w[h]`. Goal: write the index equation, then the code; the editorial gives the `matmul` form of each. Numeric, tolerance for float32 sums in a different order. Mutants: scores transposed to `(B, H, S, T)` (hidden when `T == S`); `bilinear` returning the `(B, B)` table's row sums; `w` transposed (hidden when `D == E`); one head's weight used for all.

**`expand-and-repeat`** — Implement `repeat_kv_heads(k, n_rep)` mapping `(B, G, T, D)` to `(B, G * n_rep, T, D)` with head `g * n_rep + r` equal to group `g`; `expand_for_beams(x, K)` mapping `(B, T)` to `(B * K, T)` with row `b * K + j` equal to row `b`; and `tile_positions(pos, B)` mapping `(T,)` to `(B, T)`. Goal: repeating each element (interleave) against repeating the whole (tile), and `expand` as the copy-free form. Numeric. Mutants: heads tiled (`0, 1, 0, 1` for `0, 0, 1, 1`); beams tiled; the repeat along `T`; positions repeated into `(T, B)` and reshaped.

**`join-pad-split`** — Implement `pad_rows(rows, pad, side)` mapping a list of 1-D integer tensors to `(ids, mask)`, an integer `(B, T_max)` batch padded on the given side and its boolean mask of real tokens; `shift_right(ids, bos)` mapping `(B, T)` to `(B, T)` with `bos` first and the last column dropped; and `stack_steps(steps)` mapping a list of `T` tensors of shape `(B, V)` to `(B, T, V)`. Goal: `cat` joins along an axis that exists, `stack` makes a new one, and where padding goes. Numeric, with rows of equal length and a batch of one row. Mutants: padding on the other side; the mask inverted; the shift by `roll` (the last token wraps to the front); the stack on axis 0; `cat` for `stack`.

**`dtypes-and-promotion`** — Implement `dequantize(q, scale, zero_point)` mapping int8 `q` to float32 `(q - zero_point) * scale` for `zero_point` in `[-128, 127]`; `saturating_add_u8(a, b)` adding two uint8 tensors with results above 255 clamped, returned as uint8; and `block_and_offset(pos, size)` mapping integer positions that may be negative to `(pos // size, pos % size)` with floor rounding. Goal: integer arithmetic wraps in the narrow type, so widen before the operation that can overflow, and know which division a framework performs. Numeric, exact. Mutants: the subtraction in int8; the uint8 addition that wraps; the clamp after the wrap; truncating division; a remainder with the sign of the dividend.

**`loops-to-tensors`** — Implement, without a Python loop over rows or elements, `segment_means(x, seg, S)` mapping `(N, D)` and segment ids `(N,)` in `[0, S)` to the `(S, D)` mean of each segment (zero for an empty one); `common_prefix_len(a, b)` mapping two `(B, T)` id tensors to the `(B,)` length of each row pair's common prefix; and `repeated_bigram_ban(ids, V)` mapping `(B, T)` to the boolean `(B, V)` mask of tokens `v` such that the pair (last token of the row, `v`) already occurs in the row. Goal: put the earlier drills together (adding scatter, running product, comparison and scatter). Numeric; speed relative to the reference at a size where the row loop is at least 5x slower in every framework, measured by the author (if it is not, the problem keeps the numeric judge alone and says so). Mutants: the mean over all `N`; NaN for an empty segment; a prefix length that counts matches after the first mismatch; the ban that includes the final position's own successor slot (reads past the row) or bans the last token itself; the row loop (correct, too slow).

**Batches.** A: orders 10 to 50. B: orders 60 to 100. C: orders 110 to 150. Each batch is
authored, then reviewed (process step 6), then committed.

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

Modern architectural components (section 9), checked 2026-11-21
- Installed reference implementations, read as source code (paths under `python/.venv/lib/python3.15/site-packages/`). Verified by reading, not by loading any model.
  - `transformers` 5.23.0: `models/qwen3_5/modeling_qwen3_5.py` and `configuration_qwen3_5.py` (gated delta rule in step and chunked form, gated norm, causal convolution state, gated attention, zero-centered RMSNorm, partial RoPE, layer schedule); `models/glm_moe_dsa/modeling_glm_moe_dsa.py` and `configuration_glm_moe_dsa.py` (indexer, top-k mask, shared indices, sigmoid router, dense-first schedule); `models/deepseek_v4/modeling_deepseek_v4.py` (HCA and CSA compressors and caches, shared key-value attention with output rotation, hyper-connections, `sqrtsoftplus` and hash routers; the model's forward only, not its configuration file); `models/gemma4/modeling_gemma4.py` and `configuration_gemma4.py` (per-layer embeddings, per-type RoPE, KV sharing modes, router and parallel expert branch); `models/nemotron_h/modeling_nemotron_h.py` (Mamba-2 mixer and one-token update, grouped gated norm, latent projection around experts); `models/lfm3/modeling_lfm3.py` (`Lfm3ShortConv` and the layer's norms only); `models/diffusion_gemma/generation_diffusion_gemma.py` (sampler, stopping criteria, denoising step, generate loop; the denoiser's modeling file was only searched for self-conditioning); `modeling_rope_utils.py` (`_compute_yarn_parameters`, `_compute_proportional_rope_parameters`); `modeling_layers.py` (`MtpLayer`, `MtpModel`); `generation/candidate_generator.py` (`MTPCandidateGenerator`, `DFlashTokenCandidateGenerator`); `utils/output_capturing.py` (the last hidden state is the normed one); `activations.py` (`SqrtSoftplusActivation`).
  - `mlx-lm` 0.32.2: `models/gated_delta.py`, `models/qwen3_5.py`, `models/qwen3_next.py` (attention, gated norm, expert block), `models/mamba2.py`, `models/ssm.py`, `models/kimi_k3.py` (expert selection, SiTU, attention residual mixing, latent experts, layer wiring; its delta and latent attention classes were not read).
  - Not read although present: `transformers` `inkling2`, `mistral_large4`, `mimo_v2_flash`, `qwen4_exp`, `kimi_linear`, `gpt_oss`, `mamba2`; `mlx-lm` `deepseek_v41.py`, `gemma4_text.py`, `kimi_linear.py`.
- Model files fetched from Hugging Face (`config.json` unless noted; no weight file was downloaded): https://huggingface.co/Qwen/Qwen3.5-0.8B (also `model.safetensors.index.json` and the model API record: revision, license, sizes, parameter counts), https://huggingface.co/Qwen/Qwen3.5-0.8B-Base , https://huggingface.co/Qwen/Qwen3.5-2B , https://huggingface.co/zai-org/GLM-5.3 , https://huggingface.co/google/gemma-4-E2B-it , https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813 , https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 , https://huggingface.co/moonshotai/Kimi-K3.1-Pro , https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 , https://huggingface.co/LiquidAI/LFM3-1.7B , https://huggingface.co/google/diffusiongemma-26B-A4B-it , https://huggingface.co/openai/gpt-oss-20b . Not verified: any license other than Qwen3.5's metadata; any model card text; any tensor shape or dtype inside a weight file.
- Papers. Equations read: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", https://arxiv.org/abs/2412.06464 (Eq. 10, block design, chunkwise form; first 100,000 characters of the HTML); "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models", https://arxiv.org/abs/2512.02556 (Eq. 1 and 2); "DeepSeek-V3 Technical Report", https://arxiv.org/abs/2412.19437 (Eq. 12 to 16 and 21 to 23). Abstract or title page only: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", https://arxiv.org/abs/2606.19348 ; "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse", https://arxiv.org/abs/2603.12201 ; "Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking", https://arxiv.org/abs/2505.24857 ; Dao and Gu, "Transformers are SSMs", https://arxiv.org/abs/2405.21060 ; Peng et al., "YaRN", https://arxiv.org/abs/2309.00071 . Not read: the Gemma 4 report (arXiv 2607.02770), the hyper-connections paper the DeepSeek code cites as "Xie et al., 2026", any Kimi, Qwen3.5, Nemotron, or LFM architecture paper. Author names above are as the arXiv pages or the code give them; verify before citing in an editorial.
- Measured on this machine (MLX 0.32.4): `mx.linalg.solve_triangular` raises on the GPU stream and works with `stream=mx.cpu`. Checked numerically in float64 with a throwaway script: the chunked delta rule of section 9.2 equals the step form; the state-space matrix form equals the step loop; Sinkhorn started on rows or on columns agrees to 2e-14 after 20 iterations and differs by 0.02 after 2.
- Starting inventory: the 2026-11-20 architecture survey (configs and cards only). Treated as leads; its slips found here are listed in section 9.8.
