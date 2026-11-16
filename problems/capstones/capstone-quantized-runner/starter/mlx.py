import mlx.core as mx

from whetstone.kit import DecoderConfig, ops

Checkpoint = dict[str, mx.array]
Cache = list[tuple[mx.array, mx.array]]  # per layer: (keys, values)


def quantize_checkpoint(params: dict[str, mx.array], bits: int, group_size: int) -> Checkpoint:
    """Stage 1. Quantize every matrix of `params` to `bits` (4 or 8) bits per weight.

    For a 2-D tensor `P.weight` of shape (out, in) the checkpoint holds
      `P.weight`: uint8, (out, in * bits / 8), the packed codes,
      `P.scales`, `P.biases`: float32, (out, in / group_size).
    Every other tensor is copied unchanged. Nothing else is stored.
    """
    raise NotImplementedError


def forward_q(ckpt: Checkpoint, cfg: DecoderConfig, tokens: mx.array, group_size: int) -> mx.array:
    """Stage 2. Logits (B, T, vocab_size) for token ids (B, T), computed from the codes."""
    raise NotImplementedError


def step_q(
    ckpt: Checkpoint,
    cfg: DecoderConfig,
    tokens: mx.array,
    cache: Cache | None,
    group_size: int,
) -> tuple[mx.array, Cache]:
    """Stage 3. Run T new tokens (B, T) that continue the S tokens of `cache`.

    Returns `(logits, new_cache)`: logits (B, T, vocab_size) for the new tokens, and
    one `(keys, values)` pair per layer, each (B, n_kv_heads, S + T, head_dim), keys
    after their RoPE rotation. `cache` is None when nothing came before (S = 0) and is
    never modified.
    """
    raise NotImplementedError


def generate_q(
    ckpt: Checkpoint,
    cfg: DecoderConfig,
    prompt: list[int],
    max_new_tokens: int,
    group_size: int,
    cache: Cache | None = None,
) -> list[int]:
    """Stage 3. Greedy continuation of `prompt` (which continues `cache`, if given).

    Returns `max_new_tokens` Python ints. Every token goes through the model once.
    """
    raise NotImplementedError


def bits_per_weight(ckpt: Checkpoint, group_size: int) -> float:
    """Stage 4. Stored bits of the checkpoint divided by the model's parameter count."""
    raise NotImplementedError


def quantize_mixed(params: dict[str, mx.array], group_size: int, max_bpw: float) -> Checkpoint:
    """Stage 4. 8 bits for the embedding and the head, 4 bits for every other matrix.

    Raises ValueError if the checkpoint would exceed `max_bpw` bits per weight.
    """
    raise NotImplementedError
