"""The kit: supporting objects that problems hand to solutions.

    from whetstone.kit import TinyLM

Kit objects are small and meant to be read. Anything with instrumentation exposes a
``counters`` attribute for the cost judge.

    models       TinyLM, KVCache, draft_target_pair, LayeredLM, LayerCache, RecurrentLM,
                 RecurrentState, TinyMoE, CountingExperts, moe_route_ref, moe_dense_ref,
                 TinyEncoder, CountingScorer
    weights      DecoderConfig, decoder_params, layer_params, decoder_forward_ref,
                 decoder_block_ref
    tensor ops   ops (softmax, rms_norm, rope, attention, swiglu, sampler filters)
    text         ToyVocab, CharDFA, DEAD, TokenConstraint, DFAConstraint, Grammar
    serving      SimModel, SimEngine, CostModel, SimRequest, workloads, ScriptedLM,
                 SimCluster, Replica, Outcome, ClusterMetrics, ClusterError
    agents       ScriptedChatModel, ScriptedTool, reply, tool_call, assemble,
                 ChatProtocolError, ScriptExhausted
    devices      SimDevices, OutOfDeviceMemory
    retrieval    vectors (clustered, queries_near, exact_topk, recall, CountingDistance)
    real models  RealModel, RealLM, RealConfig, Weights, SafetensorsFile, ByteLevelBPE,
                 dequantize, unpack_codes, bf16_to_f32, SMALL, LARGE_8BIT
    files        CountingFile
    numbers      logit_kl
    MLX          MLXTinyLM, EvalSpy, TraceCounter, mlx_memory, pending_ops, watch_load,
                 LazyClock
    CUDA on CPU  SyncSpy, DeviceSpy, FakeDevice, RealDevice, OOMModel, FakeCudaGraph,
                 CompileCounter, TransferSpy, tritonshim

The MLX names need the ``mlx`` extra and are imported on first use, so the kit itself
imports on any machine. The CUDA tools work on CPU tensors and import torch on first
use. Real-model objects need only numpy plus the downloaded asset.
"""

from typing import Any

from whetstone.kit import ops, tritonshim, vectors, workloads
from whetstone.kit.bpe import ByteLevelBPE
from whetstone.kit.chatmodel import (
    ChatProtocolError,
    ScriptedChatModel,
    ScriptedTool,
    ScriptExhausted,
    assemble,
    reply,
    tool_call,
)
from whetstone.kit.constraint import DFAConstraint, TokenConstraint
from whetstone.kit.counters import Counters
from whetstone.kit.cudatools import (
    CompileCounter,
    DeviceSpy,
    FakeCudaGraph,
    FakeDevice,
    OOMModel,
    RealDevice,
    SyncSpy,
    TransferSpy,
)
from whetstone.kit.decoder import (
    DecoderConfig,
    decoder_block_ref,
    decoder_forward_ref,
    decoder_params,
    layer_params,
)
from whetstone.kit.dfa import DEAD, CharDFA
from whetstone.kit.draft import draft_target_pair
from whetstone.kit.encoder import CountingScorer, TinyEncoder
from whetstone.kit.grammar import Chars, Grammar, Ref
from whetstone.kit.layeredlm import LayerCache, LayeredLM
from whetstone.kit.metrics import logit_kl
from whetstone.kit.mlxtools import (
    CountingFile,
    EvalSpy,
    LazyClock,
    TraceCounter,
    mlx_memory,
    pending_ops,
    watch_load,
)
from whetstone.kit.moe import CountingExperts, TinyMoE, moe_dense_ref, moe_route_ref
from whetstone.kit.realmodel import (
    LARGE_8BIT,
    SMALL,
    RealConfig,
    RealLM,
    RealModel,
    Weights,
    dequantize,
    unpack_codes,
)
from whetstone.kit.recurrent import RecurrentLM, RecurrentState
from whetstone.kit.safetensors import SafetensorsFile, bf16_to_f32
from whetstone.kit.scriptedlm import ScriptedLM, ScriptedStream, StreamRecord
from whetstone.kit.simcluster import ClusterError, ClusterMetrics, Outcome, Replica, SimCluster
from whetstone.kit.simdevices import OutOfDeviceMemory, SimDevices
from whetstone.kit.simengine import (
    STEP_DONE_PRIORITY,
    CostModel,
    OutOfBlocks,
    ProtocolViolation,
    SimEngine,
    StepResult,
)
from whetstone.kit.simmodel import SimModel
from whetstone.kit.tinylm import KVCache, TinyLM
from whetstone.kit.vocab import ToyVocab
from whetstone.kit.workloads import SimRequest


def __getattr__(name: str) -> Any:
    # MLXTinyLM subclasses mlx.nn.Module, so its module cannot be imported without MLX.
    if name == "MLXTinyLM":
        from whetstone.kit.mlxlm import MLXTinyLM

        return MLXTinyLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEAD",
    "LARGE_8BIT",
    "SMALL",
    "STEP_DONE_PRIORITY",
    "ByteLevelBPE",
    "CharDFA",
    "Chars",
    "ChatProtocolError",
    "ClusterError",
    "ClusterMetrics",
    "CompileCounter",
    "CostModel",
    "Counters",
    "CountingExperts",
    "CountingFile",
    "CountingScorer",
    "DFAConstraint",
    "DecoderConfig",
    "DeviceSpy",
    "EvalSpy",
    "FakeCudaGraph",
    "FakeDevice",
    "Grammar",
    "KVCache",
    "LayerCache",
    "LayeredLM",
    "LazyClock",
    "MLXTinyLM",
    "OOMModel",
    "OutOfBlocks",
    "OutOfDeviceMemory",
    "Outcome",
    "ProtocolViolation",
    "RealConfig",
    "RealDevice",
    "RealLM",
    "RealModel",
    "RecurrentLM",
    "RecurrentState",
    "Ref",
    "Replica",
    "SafetensorsFile",
    "ScriptExhausted",
    "ScriptedChatModel",
    "ScriptedLM",
    "ScriptedStream",
    "ScriptedTool",
    "SimCluster",
    "SimDevices",
    "SimEngine",
    "SimModel",
    "SimRequest",
    "StepResult",
    "StreamRecord",
    "SyncSpy",
    "TinyEncoder",
    "TinyLM",
    "TinyMoE",
    "TokenConstraint",
    "ToyVocab",
    "TraceCounter",
    "TransferSpy",
    "Weights",
    "assemble",
    "bf16_to_f32",
    "decoder_block_ref",
    "decoder_forward_ref",
    "decoder_params",
    "dequantize",
    "draft_target_pair",
    "layer_params",
    "logit_kl",
    "mlx_memory",
    "moe_dense_ref",
    "moe_route_ref",
    "ops",
    "pending_ops",
    "reply",
    "tool_call",
    "tritonshim",
    "unpack_codes",
    "vectors",
    "watch_load",
    "workloads",
]
