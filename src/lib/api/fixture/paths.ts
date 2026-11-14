import { slug } from "./catalog";

/** A fixture path before validation: stages list problem titles (or ids) from the fixture catalog. */
export interface FixturePath {
  id: string;
  title: string;
  description: string;
  stages: { title: string; description: string; problems: string[] }[];
}

/**
 * The fixture's study path: 75 problems over the fixture catalog, in the order
 * a course would teach them. Entries are problem titles as they appear in
 * `tracks.ts` (their ids are the slugged titles) or the ids of authored problems.
 */
const INFERENCE_75: FixturePath = {
  id: "inference-75",
  title: "Inference 75",
  description:
    "The shortest route through LLM inference: from the arithmetic inside one layer to the systems that serve tokens to many users at once.",
  stages: [
    {
      title: "Numerics",
      description:
        "Float32 arithmetic and the normalizations and activations every layer is built from, with the edge cases that break naive versions.",
      problems: [
        "stable-softmax",
        "log-sum-exp",
        "RMS Normalization",
        "Layer Normalization from Scratch",
        "Cross-Entropy from Logits",
        "GELU: Exact and Tanh Approximation",
        "SwiGLU Feed-Forward",
      ],
    },
    {
      title: "From text to tokens",
      description:
        "How text becomes the integer ids a model sees, and how ids become text again without losing a character.",
      problems: [
        "Byte-Level Pre-Tokenizer",
        "bpe-merge",
        "BPE Encode with a Merge Table",
        "BPE Decode and UTF-8 Repair",
        "Special Tokens and Added Vocabulary",
        "Incremental Detokenization",
        "Chat Template Renderer",
      ],
    },
    {
      title: "Attention",
      description:
        "The inner loop of the transformer: scores, masks, and positions, then the variants that make it cheaper.",
      problems: [
        "scaled-dot-product-attention",
        "causal-mask",
        "Multi-Head Split and Merge",
        "Sinusoidal Position Table",
        "rotary-embeddings",
        "Grouped-Query Attention",
        "Sliding-Window Attention",
        "Padding-Aware Batched Attention",
        "Chunked Attention Under a Memory Cap",
      ],
    },
    {
      title: "The forward pass",
      description:
        "Assembling layers into a decoder-only model whose activations match a reference, and loading real weights into it.",
      problems: [
        "Pre-Norm Residual Block",
        "Tied Embedding and Unembedding",
        "Final Norm and Head",
        "Decoder-Only Forward Pass",
        "Loading Weights by Name",
        "Parameter Count Audit",
        "Activation Checkpoint Diff",
      ],
    },
    {
      title: "Decoding",
      description:
        "Turning logits into tokens: greedy and stochastic samplers, the filters and penalties in front of them, and search.",
      problems: [
        "Greedy Decoding with Stop Sequences",
        "temperature-top-k",
        "Inverse-CDF Categorical Sampling",
        "top-p-sampling",
        "Min-P Sampling",
        "Repetition and Frequency Penalties",
        "Logit Bias and Banned Tokens",
        "Sampler Chains",
        "Beam Search with Length Penalty",
      ],
    },
    {
      title: "The KV cache",
      description:
        "Computing each key and value once: cached decoding, the layouts a cache can take, and what it costs in memory.",
      problems: [
        "kv-cache-decode",
        "Cache Append Without Reallocation",
        "Rotary Positions with a Cache",
        "Batched Cache with Ragged Lengths",
        "Prefix Cache Reuse Across Requests",
        "Cache Truncation and Rollback",
        "Paged Cache Blocks",
        "Cache Memory Accounting",
      ],
    },
    {
      title: "Smaller and faster",
      description:
        "Spending less per token: fewer bits per weight, less memory per model, and several drafted tokens verified in one pass.",
      problems: [
        "int8-affine-quant",
        "Symmetric Per-Channel Quantization",
        "Dynamic Range Calibration",
        "Four-Bit Group Quantization",
        "Quantized Matmul with Integer Accumulation",
        "Peak Memory Estimator",
        "Memory-Mapped Weight Loading",
        "Speculative Sampling Acceptance Rule",
        "Draft-and-Verify Loop",
        "Cache Rollback After Rejection",
      ],
    },
    {
      title: "Batching and serving",
      description:
        "The model as a service: sharing one accelerator between requests, then the protocol surface that streams tokens to clients.",
      problems: [
        "Static Batcher with Timeout",
        "Length-Bucketed Batching",
        "continuous-batching",
        "Token-Budget Admission Control",
        "Chunked Prefill",
        "Completions Endpoint",
        "sse-token-stream",
        "Client Disconnect Cancellation",
        "Stop Sequences Across Chunk Boundaries",
        "Token Bucket Rate Limiter",
        "Metrics: Time to First Token",
      ],
    },
    {
      title: "Constrained output and whole systems",
      description:
        "Holding generation to a grammar without slowing it down, then putting every earlier piece together into one engine and one server.",
      problems: [
        "Regex to DFA",
        "Token-Level DFA Masks",
        "JSON Schema to Grammar",
        "Constrained Sampling Renormalization",
        "Streaming Structured Output",
        "Tiny Inference Engine",
        "OpenAI-Compatible Server",
      ],
    },
  ],
};

/** The fixture paths, with every entry resolved to a problem id. */
export const FIXTURE_PATHS: FixturePath[] = [INFERENCE_75].map((path) => ({
  ...path,
  stages: path.stages.map((stage) => ({ ...stage, problems: stage.problems.map(slug) })),
}));
