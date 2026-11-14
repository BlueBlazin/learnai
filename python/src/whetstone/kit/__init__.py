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
"""

from whetstone.kit import ops, workloads
from whetstone.kit.constraint import DFAConstraint, TokenConstraint
from whetstone.kit.counters import Counters
from whetstone.kit.decoder import (
    DecoderConfig,
    decoder_block_ref,
    decoder_forward_ref,
    decoder_params,
    layer_params,
)
from whetstone.kit.dfa import DEAD, CharDFA
from whetstone.kit.draft import draft_target_pair
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

__all__ = [
    "DEAD",
    "STEP_DONE_PRIORITY",
    "CharDFA",
    "CostModel",
    "Counters",
    "DFAConstraint",
    "DecoderConfig",
    "KVCache",
    "OutOfBlocks",
    "ProtocolViolation",
    "ScriptedLM",
    "ScriptedStream",
    "SimEngine",
    "SimModel",
    "SimRequest",
    "StepResult",
    "StreamRecord",
    "TinyLM",
    "TokenConstraint",
    "ToyVocab",
    "decoder_block_ref",
    "decoder_forward_ref",
    "decoder_params",
    "draft_target_pair",
    "layer_params",
    "ops",
    "workloads",
]
