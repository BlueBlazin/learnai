"""The kit: supporting objects that problems hand to solutions.

    from whetstone.kit import TinyLM

Kit objects are small and meant to be read. Anything with instrumentation exposes a
``counters`` attribute for the cost judge.

    models       TinyLM, KVCache, draft_target_pair
    weights      DecoderConfig, decoder_params, layer_params, decoder_forward_ref,
                 decoder_block_ref
    tensor ops   ops (softmax, rms_norm, rope, attention, swiglu, sampler filters)
    text         ToyVocab, CharDFA, DEAD, TokenConstraint, DFAConstraint
    serving      SimModel, SimEngine, CostModel, SimRequest, workloads, ScriptedLM
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

from whetstone.kit import ops, tritonshim, workloads
from whetstone.kit.bpe import ByteLevelBPE
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
from whetstone.kit.safetensors import SafetensorsFile, bf16_to_f32
from whetstone.kit.scriptedlm import ScriptedLM, ScriptedStream, StreamRecord
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
    "CompileCounter",
    "CostModel",
    "Counters",
    "CountingFile",
    "DFAConstraint",
    "DecoderConfig",
    "DeviceSpy",
    "EvalSpy",
    "FakeCudaGraph",
    "FakeDevice",
    "KVCache",
    "LazyClock",
    "MLXTinyLM",
    "OOMModel",
    "OutOfBlocks",
    "ProtocolViolation",
    "RealConfig",
    "RealDevice",
    "RealLM",
    "RealModel",
    "SafetensorsFile",
    "ScriptedLM",
    "ScriptedStream",
    "SimEngine",
    "SimModel",
    "SimRequest",
    "StepResult",
    "StreamRecord",
    "SyncSpy",
    "TinyLM",
    "TokenConstraint",
    "ToyVocab",
    "TraceCounter",
    "TransferSpy",
    "Weights",
    "bf16_to_f32",
    "decoder_block_ref",
    "decoder_forward_ref",
    "decoder_params",
    "dequantize",
    "draft_target_pair",
    "layer_params",
    "logit_kl",
    "mlx_memory",
    "ops",
    "pending_ops",
    "tritonshim",
    "unpack_codes",
    "watch_load",
    "workloads",
]
