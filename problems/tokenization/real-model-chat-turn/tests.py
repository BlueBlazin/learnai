"""Tests for real-model-chat-turn.

Every expected prompt and every expected reply comes from `fixtures/make.py`: the
prompt text is the model's own Jinja template rendered by `transformers`, the prompt
ids are those of `tokenizers`, and the real replies are `transformers`' greedy
`generate` in float32. The reference solution is never the oracle.

- The cases marked `assets=[]` use a 382-token miniature of Granite's tokenizer
  (`fixtures/mini-tokenizer.json`) and a scripted model: `step` answers with logits
  whose largest entry is the next token of a script, whatever it was fed. What was fed
  is recorded and compared exactly.
- The other cases use the real tokenizer and the real granite-4.0-350m through the
  kit's `RealLM`, with its KV cache behind `step`.

Three things are checked in every case, in this order: the calls of `step` (the
prompt, then one generated token per call), the yielded pieces, and that each piece
was yielded before `step` was called again.
"""

import codecs

import numpy as np

from learnai.judge import case, check
from learnai.kit import ByteLevelBPE, RealModel

ASSET = "granite-4.0-350m"
REVISION = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
STOP = "<|end_of_text|>"

_cache: dict[str, object] = {}


class Step:
    """The `step` handed to the solution. Records every call.

    `model(call_index, token_ids)` returns the logits after that call.
    """

    def __init__(self, model):
        self.model = model
        self.calls: list[list[int]] = []

    def __call__(self, token_ids):
        if not isinstance(token_ids, list) or not token_ids:
            raise TypeError("step: token_ids must be a non-empty list of ints")
        if not all(type(t) is int for t in token_ids):
            raise TypeError("step: token_ids must be a list of Python ints")
        self.calls.append(list(token_ids))
        return self.model(len(self.calls) - 1, token_ids)


def scripted(script, vocab_size, stop):
    """A model that emits `script`, one token per call, and then stops forever."""

    def model(call_index, token_ids):
        target = script[call_index] if call_index < len(script) else stop
        # Unequal background logits; the stop token is the runner-up, so an argmax
        # that is off by anything shows.
        logits = (np.arange(vocab_size, dtype=np.float32) * 7.0) % 3.0 - 6.0
        logits[stop] = 2.0
        logits[target] = 5.0
        return logits

    return model


def real_model(lm):
    """The real model behind `step`: the kit's forward pass, continuing its own cache."""
    state = {"cache": None}

    def model(call_index, token_ids):
        logits, state["cache"] = lm(token_ids, state["cache"])
        return np.asarray(logits[-1], dtype=np.float32)

    return model


def expected_arrivals(tokenizer, generated):
    """For each expected piece, the number of `step` calls made when it must arrive.

    The piece completed by generated token i is yielded after i + 1 calls (the prompt
    and the i tokens before it); the final U+FFFD arrives after the last call.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    special = set(tokenizer.specials.values())
    arrivals = []
    for i, token in enumerate(generated):
        if decoder.decode(b"" if token in special else tokenizer.token_bytes(token)):
            arrivals.append(i + 1)
    if decoder.decode(b"", final=True):
        arrivals.append(None)  # after the last call, however many there were
    return arrivals


def run(ctx, tokenizer, model, item, *, hidden):
    step = Step(model)
    out = ctx.solution.chat_turn(
        [dict(m) for m in item["messages"]], tokenizer, step, item["max_new_tokens"]
    )
    check.that(
        hasattr(out, "__next__"),
        "chat_turn must be a generator (or return an iterator) of text pieces",
        f"got: {type(out).__name__}",
    )
    pieces, calls_at_yield = [], []
    for piece in out:
        pieces.append(piece)
        calls_at_yield.append(len(step.calls))
        check.that(len(pieces) <= 200, "chat_turn: yielded more than 200 pieces")

    # 1. The calls of step: the whole prompt, then one generated token per call.
    check.that(len(step.calls) > 0, "chat_turn: step was never called")
    prompt = item["prompt_ids"]
    if step.calls[0] != prompt:
        got_text = tokenizer.decode(step.calls[0])
        check.that(
            False,
            f"the first call of step received {len(step.calls[0])} token ids that are not "
            "the ids of the rendered prompt",
            f"expected the ids of:\n{item['prompt_text']!r}\n"
            f"got ids that decode to:\n{got_text!r}\n"
            f"expected ids: {prompt}\ngot ids:      {step.calls[0]}",
        )
    generated = item["generated"]
    # A reply ended by the stop token has fed every generated token; one ended by the
    # limit has not fed its last token.
    fed = generated if item["stopped"] else generated[:-1]
    check.equal(
        [list(call) for call in step.calls[1:]],
        [[token] for token in fed],
        label="the calls of step after the prompt (one generated token each)",
    )

    # 2. The text.
    check.that(
        all(isinstance(p, str) and p for p in pieces),
        "chat_turn: every yielded piece must be a non-empty str",
        f"got: {pieces!r}",
    )
    check.equal(pieces, item["pieces"], label="the yielded pieces")

    # 3. Streaming: a piece is yielded before the model is called again.
    late = [
        index
        for index, (got, want) in enumerate(
            zip(calls_at_yield, expected_arrivals(tokenizer, generated), strict=True)
        )
        if want is not None and got > want
    ]
    check.that(
        not late,
        f"{len(late)} of {len(pieces)} pieces were yielded only after step had been called "
        f"again (first: piece {late[0] if late else 0})",
        "yield the text of a token before calling step with that token",
    )
    if not hidden:
        ctx.metric("generated_tokens", len(generated), "tokens")
    return pieces


def mini(ctx, name, *, hidden=True):
    if "mini" not in _cache:
        _cache["mini"] = ByteLevelBPE.from_json(ctx.fixture("mini-tokenizer.json"))
    tokenizer = _cache["mini"]
    item = next(c for c in ctx.fixture("mini-cases.json")["cases"] if c["name"] == name)
    stop = tokenizer.specials[STOP]
    model = scripted(item["script"], tokenizer.vocab_size, stop)
    return run(ctx, tokenizer, model, item, hidden=hidden)


def granite(ctx, name, *, hidden=True):
    fixture = ctx.fixture("granite-4.0-350m.json")
    check.that(fixture["revision"] == REVISION, "the fixture was made from another revision")
    if "real" not in _cache:
        _cache["real"] = RealModel(ctx.asset(ASSET))
    model = _cache["real"]
    item = next(c for c in fixture["cases"] if c["name"] == name)
    return run(ctx, model.tokenizer, real_model(model.lm()), item, hidden=hidden)


# ---------------------------------------------------------------------------------
# A miniature tokenizer and a scripted model: no download needed.
# ---------------------------------------------------------------------------------


@case("the worked example: 'Hi!' with the default system turn; the reply ends in an emoji",
      public=True, assets=[])  # fmt: skip
def worked_example(ctx):
    pieces = mini(ctx, "example", hidden=False)
    check.equal("".join(pieces), "Hello! 🙂", label="the whole reply")


# Rejects first_system_twice.
@case("a system message first, then two user turns around an assistant turn",
      public=True, assets=[])  # fmt: skip
def system_first(ctx):
    mini(ctx, "system_first", hidden=False)


# Rejects empty_system_kept and strips_content.
@case("an empty system message, a later system message, content with spaces and newlines",
      public=True, assets=[])  # fmt: skip
def empty_system(ctx):
    mini(ctx, "empty_system", hidden=False)


# Rejects drops_unfinished_character, one_token_too_many, feeds_last_token.
@case("max_new_tokens = 3 ends the reply in the middle of an emoji", public=True, assets=[])
def cut_in_character(ctx):
    mini(ctx, "cut_in_character", hidden=False)


# Rejects specials_as_text and stops_on_any_special.
@case("the model emits special tokens other than the end of the turn, and <think>",
      public=True, assets=[])  # fmt: skip
def special_tokens(ctx):
    mini(ctx, "special_tokens", hidden=False)


@case("the conversation starts with an assistant message", assets=[])
def assistant_first(ctx):
    mini(ctx, "assistant_first")


@case("the limit is reached while the model would go on", assets=[])
def limit_reached(ctx):
    mini(ctx, "limit_reached")


@case("the limit equals the length of the reply; and a limit of one token", assets=[])
def limit_edges(ctx):
    mini(ctx, "limit_is_reply_length")
    mini(ctx, "one_token")


@case("the first token is the end of the turn", assets=[])
def immediate_stop(ctx):
    mini(ctx, "immediate_stop")


# Rejects decodes_tokens_separately in more ways than the example does.
@case("invalid and unfinished byte sequences in the reply", assets=[])
def invalid_bytes(ctx):
    mini(ctx, "invalid_bytes")


@case("three-byte characters spread over two and three tokens", assets=[])
def multibyte_text(ctx):
    mini(ctx, "multibyte_text")


# ---------------------------------------------------------------------------------
# granite-4.0-350m: the real tokenizer, the real weights.
# ---------------------------------------------------------------------------------


# Any real case rejects ids_as_uint16: <|start_of_role|> is token 100264.
@case("Granite 4.0 350M: 'Reply with three different emoji and nothing else.'", public=True)
def real_three_emoji(ctx):
    granite(ctx, "three_emoji", hidden=False)


@case("Granite 4.0 350M: a translation into Japanese, with a system message")
def real_translator(ctx):
    granite(ctx, "translator")


@case("Granite 4.0 350M: the second turn of a conversation")
def real_second_turn(ctx):
    granite(ctx, "second_turn")


@case("Granite 4.0 350M: two system messages")
def real_two_system_messages(ctx):
    granite(ctx, "two_system_messages")


@case("Granite 4.0 350M: a limit of 5 tokens ends the reply inside an emoji")
def real_cut_in_emoji(ctx):
    granite(ctx, "cut_in_emoji")
