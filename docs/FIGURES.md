# Figures: the plan for the catalog

Which of the first 237 problems get which figure (the 15 problems of the `tensors` module, added on 2026-11-27, are not in this plan; two of them have a mechanism figure). The rules are in `PROBLEM_AUTHORING.md`
under "Figures". "Mechanism" here also covers a static layout, timeline, or plot.

Status on 2026-11-22: every batch below is drawn and placed.

- Maps: 9 drawings, shown by 118 problems (foundations uses the transformer drawing).
- Mechanism figures, including layouts, timelines, and plots: 86. Stepped: 15.
- Interactive: 5 components and one variant, used by 7 problems.
- 63 problems have no figure and are not listed.

A map-only row names the highlighted part. Otherwise "what it shows" describes the
other figure, and the map part is in the list above the table.

## foundations

**Map.** The transformer drawing, copied: same blocks and roles as
`tools/figures/maps/transformer.json`.
Highlights: Softmax: `stable-softmax`. The three RMSNorm: `rmsnorm`,
`half-precision-rmsnorm`. SwiGLU MLP: `gelu-silu-activations`. Attention:
`split-merge-heads`, `attention-mask-builder`. Linear: `last-token-gather`.

| problem | kind | what it shows | note |
|---|---|---|---|
| `stable-softmax` | map | Softmax | |
| `rmsnorm` | map | the three RMSNorm blocks | |
| `gelu-silu-activations` | map | SwiGLU MLP | |
| `split-merge-heads` | map, mechanism | A `(T, D)` grid cut into `H` column bands, then stacked as `(H, T, D/H)` | |
| `attention-mask-builder` | map, mechanism | `T × T` mask of a left-padded row: causal triangle, hidden padding columns, padding diagonal | visible cells `attn` |
| `last-token-gather` | map | Linear | |
| `bf16-decode` | mechanism | The 32 bits of a float32: sign, exponent, mantissa; the kept 16; the rounding bit | |
| `half-precision-rmsnorm` | map | the three RMSNorm blocks | |
| `safetensors-parse` | mechanism | File strip: length, JSON header, data, offsets | keeps `file-layout.svg` |
| `real-model-perplexity` | mechanism | Three windows over seven tokens, scored and context tokens | keeps `windows.svg` |

## tokenization

**Map.** Text to ids and back, bottom to top, no block has a role: `Trim to context`,
`Chat template`, `Split specials`, `Encode`, `Pad batch`, `Model`, `Detokenize`.
"messages" below, "text" above.
Highlights: Encode: `bpe-encode`, `bpe-encode-fast`, `wordpiece-encode`,
`unigram-viterbi`. Detokenize: `utf8-boundaries`, `streaming-detokenizer`. Split
specials, Encode, Detokenize: `tokenizer-json-load`. Whole: `real-model-chat-turn`.
The rest by name. `bpe-learn-merges` has no place.

| problem | kind | what it shows | note |
|---|---|---|---|
| `bpe-encode` | map, stepped | One word as bytes; a frame per merge in rank order; the ids | 4 frames |
| `utf8-boundaries` | map | Detokenize | |
| `pad-batch-left` | map | Pad batch | |
| `special-tokens-split` | map | Split specials | |
| `streaming-detokenizer` | map, stepped | Token bytes arriving, the held-back buffer, emitted text; a 3-byte character split over two tokens | 4 frames |
| `chat-template-render` | map | Chat template | |
| `wordpiece-encode` | map | Encode | |
| `context-window-trim` | map | Trim to context | |
| `bpe-encode-fast` | map | Encode | |
| `unigram-viterbi` | map, mechanism | Lattice over character positions: an edge per piece with its score, best path marked | |
| `tokenizer-json-load` | map | Split specials, Encode, Detokenize | |
| `real-model-chat-turn` | map | whole | |

## transformer

**Map.** Exists (`transformer.json`); `pre-norm-block` is placed. Add the other 18.
Highlights: Attention: `causal-self-attention`, `grouped-query-attention`,
`sliding-window-attention`, `attention-sinks-softmax`, `online-softmax-attention`.
Attention and RoPE: `qk-norm-attention`, `padded-batch-invariance`. RoPE: `rope`,
`rope-layout-conversion`. SwiGLU MLP: `swiglu-mlp`, `moe-router-forward`,
`moe-sparse-dispatch`. Attention and SwiGLU MLP: `fused-projection-split`. Linear:
`last-position-head`. Token embedding and Linear: `embedding-scale-softcap`. The
panel: `real-model-block`. Whole: `decoder-forward`, `real-model-logits`.

| problem | kind | what it shows | note |
|---|---|---|---|
| `causal-self-attention` | map, mechanism | MatMul, scale, mask, softmax, MatMul with shapes | keeps `attention-steps.svg` |
| `rope` | map, mechanism, interactive | Each pair turned by its own angle per position | keeps `pair-rotation.svg`; `rope-shift` is done |
| `swiglu-mlp` | map, mechanism | `x` into gate and up branches, SiLU on gate, product, down; shapes on wires | |
| `grouped-query-attention` | map, mechanism | 8 query heads over 2 KV heads: each KV head feeds 4 consecutive query heads | |
| `pre-norm-block` | map, mechanism | Residual stream with two normalized branches | done |
| `decoder-forward` | map | whole | |
| `last-position-head` | map | Linear | |
| `sliding-window-attention` | map | Attention | |
| `qk-norm-attention` | map | Attention, RoPE | |
| `padded-batch-invariance` | map | Attention, RoPE | |
| `moe-router-forward` | map, mechanism | Router, softmax, top-k, renormalize; `k` of `E` experts side by side; weighted sum | |
| `embedding-scale-softcap` | map | Token embedding, Linear | |
| `fused-projection-split` | map | Attention, SwiGLU MLP | |
| `rope-layout-conversion` | map, mechanism | Interleaved and half-split rows, lines between features | keeps `layouts.svg` |
| `attention-sinks-softmax` | map | Attention | |
| `moe-sparse-dispatch` | map, mechanism | Token rows gathered per expert, one call per expert, rows scattered back in order | |
| `online-softmax-attention` | map, stepped | Three KV blocks arrive: running max, normalizer, output; the rescale when the max grows | 4 frames |
| `real-model-block` | map, mechanism | Granite block, the differences marked | keeps `granite-block.svg`; role fills |
| `real-model-logits` | map | whole | |

## sampling

**Map.** The chain of `sampler-chain`, logits in at the bottom: `Penalties` (no
role), `Top-k`, `Top-p`, `Min-p` (embed), `Temperature` (norm), `Softmax`, `Sample`
(softmax). "token" above.
Highlights: Penalties: `repetition-penalty`, `frequency-presence-penalty`. Softmax
and Sample: `gumbel-max-sample`. Sample: `categorical-sample`. Whole:
`sampler-chain`. The filters and Temperature by name. Samplers outside this chain
(`top-n-sigma`, `xtc-sampler`, `mirostat-v2`, `dry-penalty`) are not on it.

| problem | kind | what it shows | note |
|---|---|---|---|
| `top-p-filtering` | map, mechanism | Sorted probability bars, cumulative curve, the line at `p`; the crossing token kept | plot |
| `repetition-penalty` | map | Penalties | |
| `temperature-scaling` | map | Temperature | |
| `top-k-filtering` | map | Top-k | |
| `categorical-sample` | map, mechanism | Unit interval cut by cumulative probability; `u` lands in one segment; a zero-width segment | |
| `min-p-filtering` | map | Min-p | |
| `frequency-presence-penalty` | map | Penalties | |
| `sampler-chain` | map, interactive | `sampler-order` | interactive is done; map goes first |
| `stop-sequences` | stepped | Pieces arrive; the held-back suffix; emitted text; the cut at the match | 4 frames |
| `beam-search` | stepped | Two beams, three steps: expand, prune with parent indices, `cache.select`, a finished beam leaves | 4 frames |
| `gumbel-max-sample` | map | Softmax, Sample | |
| `dry-penalty` | mechanism | History strip: an earlier sequence, its repeat at the end, match length, the penalized token | |

## kv-cache

**Map.** A cached decode loop, prompt ids in at the bottom: `Prefix lookup` (cache),
`Prefill` (no role), then a panel "per token" with `Attention` (attn), `RoPE
positions` entering it from the side (embed), `KV cache` beside it (cache) joined by
read and append wires, and `Block pool` under the cache (cache); above the panel
`Next token` (softmax) with a wire looping back into the panel.
Highlights: Prefix lookup: `prefix-cache-reuse`, `radix-prefix-cache`. Prefix lookup
and Block pool: `block-hash-prefix-cache`. Prefill: `chunked-prefill-cache`. RoPE
positions: `cached-rope-offsets`. Attention, KV cache, and their wires:
`kv-cache-append`. KV cache: `preallocated-kv-cache`, `sliding-window-cache`,
`kv-cache-quantize`, `kv-eviction-heavy-hitters`, `snapkv-compress`. Block pool:
`paged-kv-allocator`, `paged-kv-copy-on-write`. Panel, Next token, loop wire:
`batched-cached-decode`. Whole: `cached-greedy-decode`, `real-model-cached-generate`.

| problem | kind | what it shows | note |
|---|---|---|---|
| `cached-greedy-decode` | map, mechanism | Tokens fed per call: uncached rows that grow, beside cached rows of `P` then 1 | |
| `kv-cache-append` | map, stepped | Cache strip and mask over four calls | keeps `cache-steps.svg` |
| `chunked-prefill-cache` | map | Prefill | |
| `cached-rope-offsets` | map, interactive | `rope-shift` | interactive is done |
| `preallocated-kv-cache` | map, mechanism | Buffer of 8 slots before and after a write | keeps `buffer-layout.svg`; role fills |
| `batched-cached-decode` | map | panel, Next token, loop wire | |
| `prefix-cache-reuse` | map, mechanism | Stored and new prompt aligned: common prefix, the cache cut there, the tokens still fed | |
| `sliding-window-cache` | map, mechanism | Sinks and wrapped ring, returned order | keeps `ring-layout.svg`; role fills |
| `paged-kv-allocator` | map, mechanism | Block tables over six physical blocks | keeps `block-table.svg`; role fills |
| `paged-kv-copy-on-write` | map, stepped | Block pool with reference counts: fork, child append copies the shared last block, parent freed | 4 frames |
| `block-hash-prefix-cache` | map, mechanism | Two prompts as block chains; each key hashes the parent key and the block | |
| `radix-prefix-cache` | map, stepped | Tree of token edges: match, insert that splits an edge, locked path, LRU leaf evicted | 4 frames |
| `kv-cache-quantize` | map | KV cache | |
| `kv-eviction-heavy-hitters` | map | KV cache | |
| `snapkv-compress` | map | KV cache | |
| `real-model-cached-generate` | map | whole | |

## constrained

**Map.** A compile column on the left, bottom to top: `Schema to regex`, `Regex to
DFA` (no role), `Token index` (cache). On the right a panel "per token": `Model
logits` (no role), `Mask` (embed), `Pick token` (softmax), `Advance state` (cache).
`Allowed tokens` (no role) sits between Token index, Advance state, and Mask. Above
the panel `Parse tool calls` (no role).
Highlights: Allowed tokens and Advance state: `dfa-token-mask`,
`choice-trie-constraint`, `grammar-token-mask`, `structural-tag-dispatch`. The
panel: `constrained-greedy-decode`, `jump-forward-decode`. Mask:
`token-mask-apply`. The rest by name. `json-pushdown-validator` and `token-healing`
have no place.

| problem | kind | what it shows | note |
|---|---|---|---|
| `token-mask-apply` | map | Mask | |
| `choice-trie-constraint` | map, mechanism | Trie of three choices by token id, one prefixing another; EOS at complete nodes | |
| `dfa-token-mask` | map, mechanism | Automaton of `(true\|null)` with token jumps | keeps `token-jumps.svg` |
| `dfa-token-index` | map, mechanism | Vocabulary trie of the `(true\|null)` tokens, the DFA state reached at each node | same tokens as `token-jumps.svg` |
| `json-pushdown-validator` | stepped | Characters of `{"a":[1,{}]}` consumed; the container stack in each frame | 5 frames |
| `constrained-greedy-decode` | map | the panel | |
| `regex-to-dfa` | map, mechanism | `a(b\|c)*`: Thompson NFA with ε edges above, the DFA from subset construction below | |
| `json-schema-regex` | map | Schema to regex | |
| `jump-forward-decode` | map | the panel | |
| `tool-call-stream-parser` | map | Parse tool calls | |
| `grammar-token-mask` | map | Allowed tokens, Advance state | |
| `structural-tag-dispatch` | map | Allowed tokens, Advance state | |

## quantization

No map: most problems are variants of one step, rounding to codes.

| problem | kind | what it shows | note |
|---|---|---|---|
| `affine-int8` | interactive | `quant-grid-affine` | done |
| `bit-packing` | mechanism | Eight 4-bit codes in one uint32, code 0 in the lowest four bits | |
| `groupwise-4bit` | mechanism | Weight matrix `(out, in)`; groups of `group_size` along the input axis, each with scale, bias | |
| `quantized-matmul` | mechanism | `x` cut into groups; per group `x_g · codes_g` scaled, plus `bias · Σx_g`; summed | `linear` fills |
| `clipping-calibration` | interactive | `quant-grid` | done |
| `nf4-codebook` | mechanism | Normal density with the 16 NF4 levels; uniform 4-bit levels beneath | plot |
| `k-quant-superblock` | mechanism | One superblock: 8 blocks of 32 codes, 6-bit scale and minimum each, one float16 pair | |
| `gptq-column-update` | stepped | Weight matrix by columns: quantized, current, remaining; the error added to the remaining ones | 4 frames |

## speculative

**Map.** One round, verified tokens in at the bottom: `Draft` (no role), `Target,
one call` (no role), `Verify` (softmax), then on the wire that loops back `Roll back
caches` (cache) and `Adapt k` (no role).
Highlights: Draft: `prompt-lookup-draft`, `mtp-head-draft`,
`suffix-automaton-draft`. Target: `tree-attention-mask`. Verify:
`greedy-speculative-verify`, `speculative-sampling-step`,
`speculative-sampling-block`, `typical-acceptance`, `multi-draft-rejection`. Roll
back caches: `batched-speculative-accept`. Adapt k: `dynamic-draft-length`. Draft
and Target: `early-exit-self-speculation`. Whole: `speculative-decode-loop`.
`speculative-speedup-model` has no place.

| problem | kind | what it shows | note |
|---|---|---|---|
| `speculative-sampling-step` | map, interactive | `draft-overlap` | interactive is done |
| `greedy-speculative-verify` | map, mechanism | `K` draft tokens over `K+1` target argmax rows: matched prefix, first mismatch, returned token | |
| `prompt-lookup-draft` | map | Draft | |
| `speculative-sampling-block` | map | Verify | |
| `speculative-decode-loop` | map, stepped | One round: draft cache grows by `k`, target scores `k+1`, accept `n`, both caches cut | 4 frames |
| `dynamic-draft-length` | map | Adapt k | |
| `typical-acceptance` | map | Verify | |
| `batched-speculative-accept` | map | Roll back caches | |
| `mtp-head-draft` | map | Draft | |
| `tree-attention-mask` | map, mechanism | Draft tree and its mask | keeps `tree-mask.svg`; role fills |
| `multi-draft-rejection` | map | Verify | |
| `early-exit-self-speculation` | map, mechanism | Layer stack split at `exit_layer`: lower layers draft through the head, upper layers verify | |
| `suffix-automaton-draft` | map, mechanism | Suffix automaton of `abcbc`: states, transitions, dashed suffix links, occurrence counts | |

## sizing

No map: formulas and metrics with no shared drawing.

| problem | kind | what it shows | note |
|---|---|---|---|
| `latency-metrics` | mechanism | Timeline of one request: start and token times; TTFT, the ITL gaps, the TPOT span | |

## mlx

No map: habits of one API, no shared drawing.

| problem | kind | what it shows | note |
|---|---|---|---|
| `mlx-lazy-eval-placement` | mechanism | Three steps cut by `mx.eval` | keeps `eval-cuts.svg` |
| `mlx-fast-attention` | mechanism | Three mask grids, full and partly used buffer | keeps `mask-alignment.svg`; role fills |
| `mlx-compile-shapes` | mechanism | Three prompts in buckets | keeps `buckets.svg` |
| `mlx-preallocated-kv-cache` | mechanism | Buffer length against tokens appended | keeps `growth.svg` |
| `mlx-quantized-linear` | mechanism | A row of 64 weights as words, scale, bias | keeps `packed-row.svg` |
| `mlx-streaming-generate` | mechanism | GPU and Python lanes, blocking and async | keeps `overlap.svg` |
| `mlx-metal-kernel-dequant` | mechanism | One thread: word, code, group, weight | keeps `one-thread.svg` |
| `mlx-batched-generate` | mechanism | Left-padded batch and one row's mask | keeps `left-padding.svg`; role fills |

## cuda

No map: habits of one API, no shared drawing.

| problem | kind | what it shows | note |
|---|---|---|---|
| `cuda-timing-sync` | mechanism | Host and device lanes around `sync()` | keeps `async-timeline.svg` |
| `cuda-sdpa-attention` | mechanism | `is_causal` grid beside the needed mask | keeps `causal-alignment.svg`; role fills |
| `cuda-pinned-transfer` | mechanism | Copy and model lanes | keeps `prefetch-timeline.svg` |
| `cuda-memory-accounting` | mechanism | Device memory bar: tensors, split, free | keeps `allocated-reserved.svg` |
| `cuda-static-kv-cache` | mechanism | Buffer row and two mask rows | keeps `static-cache-mask.svg`; role fills |
| `cuda-graph-replay` | mechanism | Static input, graph, static output | keeps `capture-replay.svg` |
| `cuda-triton-rmsnorm` | mechanism | Memory strip with the masked tail of a block | keeps `tail-mask.svg` |
| `cuda-triton-dequant-linear` | mechanism | One program: x, codes, groups, weights by column | keeps `one-program.svg` |

## serving

**Map.** A request path, bottom to top, no block has a role: `Client`, `Router`,
then a panel "one replica" with `Endpoint`, `Rate limit`, `Admission`, `Waiting
queue`, `Engine loop`, `Token stream`. `Autoscaler` beside the Router and `Metrics`
beside the Endpoint. "tokens" above.
Highlights: Client: `retry-backoff-client`, `open-loop-load-generator`. Router:
`least-loaded-balancer`, `prefix-aware-router`. Waiting queue: `priority-scheduler`,
`vtc-fair-scheduler`, `slo-deadline-scheduler`. Engine loop: `dynamic-batcher`,
`continuous-batching-scheduler`, `chunked-prefill-scheduling`,
`preemption-recompute`. Waiting queue and Engine loop: `deadline-cancellation`.
Token stream: `sse-token-stream`, `streaming-backpressure`. The panel:
`graceful-drain`. The rest by name. `multi-lora-batching` and
`prefill-decode-disaggregation` have no place.

| problem | kind | what it shows | note |
|---|---|---|---|
| `dynamic-batcher` | map, interactive | `batch-window` | interactive is done |
| `sse-token-stream` | map | Token stream | |
| `token-bucket-limiter` | map, mechanism | Balance over time: refill slope, burst cap, grants as drops, a large request waiting | plot |
| `admission-control` | map | Admission | |
| `metrics-endpoint` | map | Metrics | |
| `continuous-batching-scheduler` | map, mechanism | One row per request over eight steps | keeps `timeline.svg` |
| `priority-scheduler` | map | Waiting queue | |
| `deadline-cancellation` | map | Waiting queue, Engine loop | |
| `chunked-prefill-scheduling` | map, mechanism | Token budget per step as stacked bars: decodes first, then chunks of a long prompt | |
| `vtc-fair-scheduler` | map, mechanism | Two tenants' counters over time; an idle tenant returns and is lifted to the minimum | plot |
| `openai-chat-endpoint` | map | Endpoint | |
| `least-loaded-balancer` | map | Router | |
| `retry-backoff-client` | map | Client | |
| `graceful-drain` | map | the panel | |
| `streaming-backpressure` | map | Token stream | |
| `open-loop-load-generator` | map, mechanism | Send times, closed loop against open loop, one slow response; latency from the intended time | |
| `slo-deadline-scheduler` | map | Waiting queue | |
| `prefix-aware-router` | map | Router | |
| `autoscaler-policy` | map | Autoscaler | |
| `preemption-recompute` | map, mechanism | Rows as in `timeline.svg`: one preempted, requeued first, recomputed, resumed | |

## systems

No map: three unrelated groups (devices, retrieval, application loops).

| problem | kind | what it shows | note |
|---|---|---|---|
| `device-map-solver` | mechanism | Devices as bars filled in module order: reserved space on device 0, unsplit group, disk | |
| `ivf-index-search` | mechanism | `ids`, `vectors`, `offsets` grouped by list | keeps `list-layout.svg` |
| `tensor-parallel-linear` | mechanism | `W_up` cut by output columns, `W_down` by input rows, across devices; one all-reduce | `linear` fills |
| `pipeline-parallel-schedule` | mechanism | Stages against time: tokens of three sequences interleaved, transfer gaps between stages | |

## frontier

No map: eight unrelated techniques.

| problem | kind | what it shows | note |
|---|---|---|---|
| `cross-layer-kv-sharing` | mechanism | Layer stack: lower layers with their own KV; upper layers read layer `share_from − 1` | `cache`, `attn` fills |
| `diffusion-unmask-step` | stepped | Masked canvas over three steps: candidates with confidence, the `n_unmask` committed each step | 4 frames |
| `hybrid-state-checkpoints` | mechanism | Token axis with a checkpoint every `interval`; the common prefix; the checkpoint resumed from | |
| `mla-latent-kv-cache` | mechanism | `x` to latent `c` and RoPE key, both cached; up-projections folded into query and output | shapes on wires |

## capstones

No map: each problem has its own architecture figure.

| problem | kind | what it shows | note |
|---|---|---|---|
| `capstone-continuous-batching-engine` | mechanism | Queue, planner, engine, allocator | keeps `architecture.svg`; role fills |
| `capstone-quantized-runner` | mechanism | Checkpoint and one step of the runner | keeps `runner.svg`; role fills |
| `capstone-speculative-engine` | mechanism | One round of the engine | keeps `engine.svg` |
| `capstone-structured-output-server` | mechanism | One request through the server | keeps `architecture.svg`; role fills |
| `capstone-real-model-runtime` | mechanism | Five stages from files to text | keeps `architecture.svg`; role fills |
| `capstone-inference-gateway` | mechanism | Admission, queue, dispatch, replicas | keeps `architecture.svg` |

## recurrent-mixers

No map: three different mixers; only the delta rule has several problems.

| problem | kind | what it shows | note |
|---|---|---|---|
| `gated-delta-step` | stepped | State `S` as a grid: decay by `g`, write `β(v − Sk)kᵀ`, read `Sq` | 3 frames; `cache` fill |
| `gated-delta-layer` | mechanism | Projection, conv over saved rows, split `q, k, v`, gates, recurrence on `S`, gated norm | both states `cache` |
| `mamba2-ssd-step` | mechanism | Lower-triangular `T × T` matrix `M`: cells fading with the decay from `s` to `t` | |
| `hybrid-decoder-forward` | mechanism | Four layers and their two kinds of state | keeps `two-kinds-of-state.svg`; role fills |

## sparse-attention

**Map.** One sparse-attention layer, `x` in at the bottom. Left branch: `Indexer
scores` (linear), `Top-k` (embed), with `Indexer keys` (cache) beside the scores. An
`idx` wire runs from Top-k to `Gather` (no role), which reads `KV cache` (cache) and
feeds `Attention` (attn).
Highlights: Indexer scores and Top-k: `indexer-topk-select`. Gather and Attention:
`gathered-sparse-attention`. Indexer keys and KV cache: `indexer-cache-decode`. The
`idx` wire: `shared-index-layers`. `compressed-kv-stream` has no place.

| problem | kind | what it shows | note |
|---|---|---|---|
| `indexer-topk-select` | map | Indexer scores, Top-k | |
| `gathered-sparse-attention` | map, mechanism | Store of `S` rows; one `idx` row with `-1` slots; only the selected rows read | |
| `indexer-cache-decode` | map, stepped | Indexer keys and KV cache growing together over three calls; the top-k rows read | 3 frames |
| `shared-index-layers` | map | the `idx` wire | |
| `compressed-kv-stream` | stepped | Rows arriving in pieces: the buffer, full windows of `rate` turned into entries, entries visible | 4 frames |

## block-components

**Map.** The transformer drawing, copied, with two changes: the MLP block is labeled
`MLP or experts`, and a `PLE` source (embed) enters a third addition after the MLP
residual.
Highlights: RoPE: `partial-rope`, `yarn-frequencies`. Attention and RoPE:
`shared-kv-rotation`. MLP or experts: `sigmoid-router-shared-expert`, `latent-moe`.
PLE and its addition: `per-layer-embeddings`. The additions and the skip wires:
`hyper-connections`.

| problem | kind | what it shows | note |
|---|---|---|---|
| `partial-rope` | map, mechanism | Which channels each function rotates and pairs | keeps `channel-map.svg` |
| `yarn-frequencies` | map, mechanism | Inverse frequency by pair index: original and YaRN curves, fast end equal, slow end divided | plot |
| `shared-kv-rotation` | map, mechanism | One pair plane: key-value turned by its position, output turned back by the query's | style of `pair-rotation.svg` |
| `sigmoid-router-shared-expert` | map | MLP or experts | |
| `latent-moe` | map | MLP or experts | |
| `per-layer-embeddings` | map | PLE and its addition | |
| `hyper-connections` | map, mechanism | `N` streams: `pre` collapses them into the block input; `post` writes back, `comb` mixes | |

## drafts-and-diffusion

No map: three problems.

| problem | kind | what it shows | note |
|---|---|---|---|
| `mtp-chain-draft` | mechanism | Two chained stages: shifted embedding and previous hidden concatenated, block, shared head, token passed on | |
| `entropy-bound-denoise-step` | mechanism | Positions as bars sorted by entropy; running sum of earlier entropies; the bound; accepted positions | plot |
| `block-diffusion-decode` | mechanism | Prompt, three canvases, prefix cache | keeps `canvas-loop.svg`; role fills |

## real-hybrid-model

No map: three problems.

| problem | kind | what it shows | note |
|---|---|---|---|
| `layer-plan-from-config` | mechanism | One cell per layer for three models | keeps `layer-plans.svg` |
| `real-hybrid-logits` | mechanism | Query projection and one rotated head | keeps `query-head-layout.svg` |

## Interactive figures

All five exist and are placed. Each is a component in `src/components/figures/`
(`sampler-order` is used by `sampler-chain`):

| name | problems | controls | what is drawn |
|---|---|---|---|
| `quant-grid` | `affine-int8`, `clipping-calibration` | bits (2 to 8), clipping fraction, symmetric or affine | Histogram of fixed samples, the grid levels, the clipped tails; clipping and rounding error as two numbers |
| `rope-shift` | `rope`, `cached-rope-offsets` | query position, key position, one shift added to both | The pairs of `q` and `k` in their planes, and the score, which moves with the difference only |
| `draft-overlap` | `speculative-sampling-step` | how far the draft `q` is from the target `p` (two sliders: sharpness, shift) | One bar per token, split into the overlap `min(p, q)`, the residual, and the rejected part of `q`; the acceptance rate and `R`, which add up to 1 |
| `batch-window` | `dynamic-batcher` | `max_batch_size`, `max_wait_ms` | The example's arrivals on a time axis, the batches they fall into, each request's wait, mean wait and batch size |

## Role fills on existing figures

Fills only; outlines and lettering stay. Mask cells follow `cache-steps.svg`: a
visible cell is `attn`, a cache slot is `cache`, a slot written by this call is
`cache` at 0.4 opacity.

| figure | fills |
|---|---|
| `transformer/real-model-block` `granite-block.svg` | norm boxes `norm`, attention `attn`, SwiGLU `mlp`, the two `× 0.263` boxes `norm` |
| `kv-cache/preallocated-kv-cache` `buffer-layout.svg` | filled slots `cache`, the two new ones at 0.4 |
| `kv-cache/sliding-window-cache` `ring-layout.svg` | buffer slots and returned keys `cache` |
| `kv-cache/paged-kv-allocator` `block-table.svg` | used token slots `cache` |
| `speculative/tree-attention-mask` `tree-mask.svg` | visible mask cells `attn` |
| `mlx/mlx-fast-attention` `mask-alignment.svg` | visible mask cells `attn` |
| `mlx/mlx-batched-generate` `left-padding.svg` | visible mask cells `attn` |
| `cuda/cuda-sdpa-attention` `causal-alignment.svg` | visible cells `attn` |
| `cuda/cuda-static-kv-cache` `static-cache-mask.svg` | buffer slots `cache` (new at 0.4), visible mask cells `attn` |
| `recurrent-mixers/hybrid-decoder-forward` `two-kinds-of-state.svg` | both kinds of state `cache`, the full-attention layer `attn` |
| `drafts-and-diffusion/block-diffusion-decode` `canvas-loop.svg` | prefix cache `cache` |
| `capstones/capstone-continuous-batching-engine` `architecture.svg` | pool blocks `cache` |
| `capstones/capstone-quantized-runner` `runner.svg` | embed box `embed`, linear-layer box `linear`, KV cache `cache` |
| `capstones/capstone-structured-output-server` `architecture.svg` | compilation cache `cache` |
| `capstones/capstone-real-model-runtime` `architecture.svg` | `sample_token` `softmax` |

The other existing figures are plots, timelines, bit and file layouts, or graphs with
no block that has a role. They stay as they are.

## Batches

Before batch 4: `tools/figures/maps.py` must accept a box without a role (seven of
the new maps have one) and a problem of another module (the foundations map).

A map counts as one figure; placing it in each listed statement is part of its batch.

| # | module | figures | mark |
|---|---|---|---|
| 1 | role fills A | the first 8 rows of the role-fill table | plain |
| 2 | role fills B | the last 7 rows | plain |
| 3 | foundations | map (copy), `split-merge-heads`, `attention-mask-builder`, `bf16-decode` | plain |
| 4 | tokenization | map, `bpe-encode`, `streaming-detokenizer`, `unigram-viterbi` | hard |
| 5 | transformer | 18 map placements, `swiglu-mlp`, `grouped-query-attention`, `moe-router-forward`, `moe-sparse-dispatch`, `online-softmax-attention` | plain |
| 6 | sampling | map, `top-p-filtering`, `categorical-sample`, `stop-sequences`, `beam-search`, `dry-penalty` | hard |
| 7 | kv-cache | map, `cached-greedy-decode`, `prefix-cache-reuse`, `paged-kv-copy-on-write`, `block-hash-prefix-cache`, `radix-prefix-cache` | hard |
| 8 | constrained | map, `choice-trie-constraint`, `dfa-token-index`, `json-pushdown-validator`, `regex-to-dfa` | hard |
| 9 | quantization | `bit-packing`, `groupwise-4bit`, `quantized-matmul`, `nf4-codebook`, `k-quant-superblock`, `gptq-column-update` | plain |
| 10 | speculative | map, `greedy-speculative-verify`, `speculative-decode-loop`, `early-exit-self-speculation`, `suffix-automaton-draft` | hard |
| 11 | serving | map, `token-bucket-limiter`, `chunked-prefill-scheduling`, `vtc-fair-scheduler`, `open-loop-load-generator`, `preemption-recompute` | hard |
| 12 | sizing, systems, frontier | `latency-metrics`, `device-map-solver`, `tensor-parallel-linear`, `pipeline-parallel-schedule`, `cross-layer-kv-sharing`, `diffusion-unmask-step`, `hybrid-state-checkpoints`, `mla-latent-kv-cache` | plain |
| 13 | recurrent-mixers, drafts-and-diffusion | `gated-delta-step`, `gated-delta-layer`, `mamba2-ssd-step`, `mtp-chain-draft`, `entropy-bound-denoise-step` | plain |
| 14 | sparse-attention | map, `gathered-sparse-attention`, `indexer-cache-decode`, `compressed-kv-stream` | hard |
| 15 | block-components | map (copy with a PLE source), `yarn-frequencies`, `shared-kv-rotation`, `hyper-connections` | plain |
| 16 | app | `quant-grid`, `rope-shift`, `draft-overlap`, `batch-window`, then their six placements | hard |

Hard for a new map design: 4, 6, 7, 8, 10, 11, 14. Batch 8 and batch 10 also draw
graphs (an NFA, a suffix automaton). Batch 16 is app code.
