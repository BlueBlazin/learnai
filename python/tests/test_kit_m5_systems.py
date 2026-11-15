"""Milestone 5 kit, systems: SimDevices, Grammar, ScriptedChatModel and ScriptedTool,
vectors, the new workloads, SimEngine's admit options, and SimCluster."""

import asyncio
import itertools
import json
import re

import numpy as np
import pytest

from whetstone.judge import SimClock, frameworks, percentile
from whetstone.kit import (
    Chars,
    ChatProtocolError,
    ClusterError,
    CostModel,
    CountingExperts,
    Grammar,
    OutOfDeviceMemory,
    Ref,
    ScriptedChatModel,
    ScriptedTool,
    ScriptExhausted,
    SimCluster,
    SimDevices,
    SimEngine,
    SimRequest,
    ToyVocab,
    assemble,
    moe_dense_ref,
    moe_route_ref,
    reply,
    tool_call,
    vectors,
    workloads,
)

COST = CostModel(base_s=0.5, prefill_s_per_token=0.125, decode_s_per_seq=0.25)


@pytest.fixture(params=["torch", "jax", "mlx"])
def fw(request):
    try:
        return frameworks.get(request.param)
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture
def sim():
    with SimClock() as clock:
        yield clock


def req(id, prompt, output, arrival=0.0, **more):
    return SimRequest(id, arrival, prompt, output, **more)


# -- SimDevices ---------------------------------------------------------------------------


def test_devices_meter_memory_and_messages(fw):
    devs = SimDevices(fw, memory_bytes=[64, 32, None])
    assert devs.n_devices == 3 and devs.used == (0, 0, 0)
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    stored = devs.put(0, "a", a)
    assert fw.is_tensor(stored) and devs.get(0, "a") is stored and devs.names(0) == ["a"]
    np.testing.assert_array_equal(fw.numpy(stored), a)
    with pytest.raises(OutOfDeviceMemory, match="over its budget of 64"):
        devs.put(0, "b", np.zeros(1, np.float32))
    assert devs.names(0) == ["a"] and devs.used == (64, 0, 0)  # the refused put stored nothing
    devs.put(0, "a", a[:2])  # replacing releases the old tensor first
    devs.put(0, "b", a[:2])
    assert devs.used == (64, 0, 0) and devs.peak == [64, 0, 0]
    devs.free(0, "b")
    devs.put(2, "big", np.zeros(10_000, np.float32))  # no budget on device 2
    assert devs.used == (32, 0, 40_000) and devs.counters.bytes_placed == 64 + 32 + 32 + 40_000
    with pytest.raises(KeyError):
        devs.get(1, "a")

    x = fw.tensor(a)
    assert devs.send(0, 1, x) is x  # the ledger moves nothing
    devs.send(0, 1, x[:1])
    devs.send(1, 2, x)
    assert devs.messages == [(0, 1, 64), (0, 1, 16), (1, 2, 64)]
    assert (devs.sent(0, 1), devs.sent(1, 0), devs.sent(1, 2)) == (2, 0, 1)
    assert devs.counters.sends == 3 and devs.counters.comm_bytes == 144
    for bad, error, message in [
        (lambda: devs.send(1, 1, x), ValueError, "does not send to itself"),
        (lambda: devs.send(0, 3, x), ValueError, r"outside \[0, 3\)"),
        (lambda: devs.send(0, "1", x), TypeError, "int index"),
        (lambda: devs.send(0, 1, [1.0]), ValueError, "expected a tensor"),
        (lambda: devs.all_reduce([x, x]), ValueError, r"one tensor per device \(3\)"),
        (lambda: devs.all_reduce([x, x, x[:1]]), ValueError, "one shape and dtype"),
        (lambda: SimDevices(fw), ValueError, "memory_bytes"),
    ]:
        with pytest.raises(error, match=message):
            bad()
    # A twin for the oracle: the stored shards, and a clean ledger.
    twin = devs.with_framework(fw)
    assert twin.names(0) == ["a"] and twin.counters.sends == 0 and twin.messages == []
    twin.put(1, "c", a[:1])
    assert devs.names(1) == []


def test_tensor_parallel_mlp_needs_exactly_one_all_reduce(fw):
    """What `tensor-parallel-linear` checks: the first layer split by output columns
    (no communication), the second by input rows with one all-reduce of the output's
    size; the all-gather variant is right and costs more."""
    rng = np.random.default_rng(0)
    d, hidden, n = 8, 32, 2
    x = rng.standard_normal((3, d)).astype(np.float32)
    w_up = rng.standard_normal((hidden, d)).astype(np.float32)  # (out, in)
    w_down = rng.standard_normal((d, hidden)).astype(np.float32)
    want = np.maximum(x @ w_up.T, 0) @ w_down.T
    half = hidden // n
    devs = SimDevices(fw, memory_bytes=[(half * d + d * half) * 4] * n)
    for i in range(n):
        devs.put(i, "up", w_up[i * half : (i + 1) * half])
        devs.put(i, "down", w_down[:, i * half : (i + 1) * half])
    assert devs.used == ((half * d + d * half) * 4,) * n
    xt = fw.tensor(x)

    megatron = devs.with_framework(fw)
    parts = []
    for i in range(n):
        h = np.maximum(np.asarray(fw.numpy(xt)) @ np.asarray(fw.numpy(megatron.get(i, "up"))).T, 0)
        parts.append(fw.tensor(h @ np.asarray(fw.numpy(megatron.get(i, "down"))).T))
    y = megatron.all_reduce(parts)
    np.testing.assert_allclose(fw.numpy(y), want, rtol=1e-4, atol=1e-5)
    assert megatron.counters.snapshot() == {
        "sends": 0, "comm_bytes": n * want.nbytes, "all_reduces": 1, "all_gathers": 0,
        "bytes_placed": 0,
    }  # fmt: skip

    gathered = devs.with_framework(fw)
    hidden_parts = [
        fw.tensor(np.maximum(x @ np.asarray(fw.numpy(gathered.get(i, "up"))).T, 0))
        for i in range(n)
    ]
    h = np.asarray(fw.numpy(gathered.all_gather(hidden_parts, axis=-1)))
    np.testing.assert_allclose(h @ w_down.T, want, rtol=1e-4, atol=1e-5)
    assert gathered.counters.all_gathers == 1
    assert gathered.counters.comm_bytes == (n - 1) * h.nbytes > megatron.counters.comm_bytes


def test_expert_parallel_dispatch_sends_each_device_its_rows_once(fw):
    """What `expert-parallel-dispatch` checks: one send and one receive per device
    with tokens, comm_bytes equal to the routed rows, the dense oracle's numbers."""
    experts = CountingExperts(fw, seed=3, n_experts=6, d_model=8, d_hidden=12)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((10, 8)).astype(np.float32)
    w_router = rng.standard_normal((6, 8)).astype(np.float32)
    placement = [0, 1, 2, 1, 2, 0]  # expert -> device; tokens start on device 0
    chosen, weights = moe_route_ref(x, w_router, 2)
    devs = SimDevices(fw, n_devices=3)
    out = np.zeros_like(x)
    routed_bytes = 0
    for device in range(3):
        mine = [e for e in range(6) if placement[e] == device]
        rows = np.unique(np.nonzero(np.isin(chosen, mine))[0])
        if not len(rows):
            continue
        here = fw.tensor(x[rows])
        if device != 0:
            here = devs.send(0, device, here)
            routed_bytes += x[rows].nbytes
        partial = np.zeros((len(rows), 8), dtype=np.float32)
        for expert in mine:
            local, slots = np.nonzero(chosen[rows] == expert)
            if len(local):
                mine_rows = fw.tensor(np.asarray(fw.numpy(here))[local])
                y = np.asarray(fw.numpy(experts(expert, mine_rows)))
                partial[local] += weights[rows[local], slots][:, None] * y
        back = fw.tensor(partial)
        if device != 0:
            back = devs.send(device, 0, back)
        out[rows] += np.asarray(fw.numpy(back))
    np.testing.assert_allclose(out, moe_dense_ref(x, w_router, experts, 2), rtol=1e-4, atol=1e-5)
    assert [devs.sent(0, d) for d in (1, 2)] == [1, 1] and [devs.sent(d, 0) for d in (1, 2)] == [
        1,
        1,
    ]
    assert devs.counters.sends == 4 and devs.counters.comm_bytes == 2 * routed_bytes
    assert experts.counters.expert_rows == 2 * 10


# -- Grammar ------------------------------------------------------------------------------

LISTS = """
# nested lists of numbers and booleans
root  ::= "[" ws (value ("," ws value)*)? "]"
value ::= [0-9]+ | "true" | "false" | root
ws    ::= " "?
"""


def test_grammar_parses_gbnf_into_rules_of_chars_and_refs():
    g = Grammar.parse(LISTS)
    assert list(g.rules) == [
        "root", "root_1", "root_2", "root_3", "root_4", "value", "value_1", "ws", "ws_1",
    ]  # fmt: skip
    assert g.root == "root" and g.source == LISTS
    bracket = Chars(((ord("["), ord("[")),))
    assert g.rules["root"][0][:2] == (bracket, Ref("ws"))
    digit = Chars(((ord("0"), ord("9")),))
    assert digit.matches("7") and digit.matches(ord("0")) and not digit.matches("a")
    # x+ is x followed by x*, and x* is `r ::= x r | (empty)`.
    assert g.rules["value"][0] == (digit, Ref("value_1"))
    assert g.rules["value_1"] == ((digit, Ref("value_1")), ())
    assert g.rules["value"][1] == tuple(Chars(((ord(c), ord(c)),)) for c in "true")
    assert g.rules["ws"] == ((Ref("ws_1"),),) and g.rules["ws_1"][1] == ()
    assert g.nullable == {"ws", "ws_1", "root_2", "root_4"} | {"value_1"}
    assert g.left_recursive == frozenset()
    assert repr(digit) == "Chars([0-9])" and "9 rules" in repr(g)


def test_grammar_notation_details_from_the_llama_cpp_documentation():
    g = Grammar.parse(
        "root ::= item{2,3} sep? tail\n"
        'item ::= [^a-c\\n] | "\\x41\\u00e9\\U0001F600"  # escapes\n'
        'sep ::= "-"{2} |\n'
        '        ";"\n'  # a line break directly after | continues the rule
        "tail ::= (\n"
        '  "." | "!"\n'  # and so do line breaks inside parentheses
        ")+ check-mate\n"
        "check-mate ::= [#+-]\n"
    )
    negated = g.rules["item"][0][0]
    assert negated.negated and negated.ranges == ((10, 10), (97, 99))
    assert negated.matches("z") and negated.matches("é") and not negated.matches("b")
    assert negated.overlaps(98, 100) and not negated.overlaps(97, 99) and negated.overlaps(0, 5)
    assert [s.ranges[0][0] for s in g.rules["item"][1]] == [0x41, 0xE9, 0x1F600]
    assert g.rules["check-mate"] == (
        (Chars(((35, 35), (43, 43), (45, 45))),),
    )  # a final `-` is literal
    assert g.accepts("xyz--.!#") and g.accepts("xAé\U0001f600;.+") and g.accepts("xy!-")
    assert not g.accepts("x.+") and not g.accepts("xyzx.+") and not g.accepts("xy.")
    # {m,n}: m copies, then n - m nested optional ones.
    assert g.rules["root"][0][:3] == (Ref("item"), Ref("item"), Ref("root_1"))
    assert g.rules["root_1"] == ((Ref("item"),), ())
    exact = Grammar.parse('root ::= "ab"{2} [x]{0,2} "c"{1,}')
    assert [s for s in ("abab", "ababc", "ababxxccc", "abc", "ababxxxc") if exact.accepts(s)] == [
        "ababc", "ababxxccc",
    ]  # fmt: skip
    assert Grammar.parse('root ::= ""').accepts("") and Grammar.parse("root ::= [a]{0}").accepts("")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "the grammar is empty"),
        ('value ::= "a"', "no `root` rule"),
        ("root ::= item", "uses `item`, which is not defined"),
        ('root ::= "a"\nroot ::= "b"', "line 2: rule `root` is defined twice"),
        ('root ::= "a" root', "can never finish"),
        ('root ::= "a" .*', "`.` is not supported"),
        ("root ::= <[1000]>", "token references"),
        ('root ::= "a" !<think>', "token references"),
        ('root ::= "abc', "string literal is not closed"),
        ("root ::= [abc", "character set is not closed"),
        ('root ::= ("a"', "group is not closed"),
        ('root ::= * "a"', "nothing to repeat"),
        ('root ::= "a"{2,1}', "bounds backwards"),
        ('root ::= "a"{x}', "written {m}, {m,}, or {m,n}"),
        ('root ::= "a"{101}', "above 100"),
        ('root ::= "\\q"', "unknown escape"),
        ('root ::= "\\x4"', "needs 2 hexadecimal digits"),
        ("root ::= [z-a]", "backwards"),
        ("root ::= []", "matches nothing"),
        ('Root ::= "a"', "expected a rule name"),
        ('root = "a"', "expected `::=`"),
        ('root ::= "a" )', r"unexpected '\)'"),
    ],
)
def test_grammar_refuses_what_is_outside_the_subset(text, message):
    with pytest.raises(ValueError, match=message):
        Grammar.parse(text)


def test_grammar_oracle_agrees_with_a_regular_expression_on_every_short_string():
    g = Grammar.parse('root ::= ("a" | "b" root "c")* tail?\ntail ::= [ab]{1,2} "c"')

    # The same language, written out: root is (a | b root c)* (tail)?, which nests.
    def member(s):
        @__import__("functools").cache
        def root(i, j):  # can s[i:j] be derived from root?
            if any(star(i, m) and re.fullmatch(r"[ab]{1,2}c", s[m:j]) for m in range(i, j)):
                return True
            return star(i, j)

        @__import__("functools").cache
        def star(i, j):
            if i == j:
                return True
            if s[i] == "a" and star(i + 1, j):
                return True
            return s[i] == "b" and any(
                root(i + 1, m) and s[m] == "c" and star(m + 1, j) for m in range(i + 1, j)
            )

        return root(0, len(s))

    strings = ["".join(t) for n in range(8) for t in itertools.product("abc", repeat=n)]
    members = {s for s in strings if member(s)}
    assert {s for s in strings if g.accepts(s)} == members and len(members) > 50
    # A prefix is anything some member starts with (members up to length 7 witness
    # every prefix up to length 4).
    prefixes = {m[:i] for m in members for i in range(len(m) + 1)}
    short = [s for s in strings if len(s) <= 4]
    assert {s for s in short if g.accepts_prefix(s)} == {s for s in short if s in prefixes}


def test_grammar_oracle_handles_left_recursion_ambiguity_and_bytes():
    g = Grammar.parse(
        "root ::= expr\n"
        'expr ::= expr "+" term | term\n'
        'term ::= [0-9]{1,3} | "(" expr ")" | "€" | "😀"\n'
    )
    assert g.left_recursive == {"expr"}
    assert g.accepts("1+(22+333)+€") and not g.accepts("1234") and not g.accepts("1+")
    assert g.accepts_prefix("1+(") and g.accepts_prefix("") and not g.accepts_prefix("1+)")
    euro, smile = "€".encode(), "😀".encode()
    assert g.accepts(euro) and g.accepts(b"1+" + smile) and g.accepts(bytearray(euro))
    # A token may end inside a character: still a prefix if some accepted character
    # starts with those bytes.
    assert g.accepts_prefix(euro[:1]) and g.accepts_prefix(euro[:2]) and g.accepts_prefix(smile[:3])
    assert not g.accepts(euro[:2])
    assert not g.accepts_prefix(b"1+\xe2\x83")  # U+20C0..U+20FF: no such term
    assert not g.accepts_prefix(b"1\xe2")  # a term cannot follow a digit
    assert not g.accepts_prefix(b"\xff") and not g.accepts_prefix(b"\xe2\x28\xa1")
    assert not g.accepts_prefix(b"\xc0\x80") and not g.accepts_prefix(
        b"\xed\xa0"
    )  # overlong, surrogate
    ambiguous = Grammar.parse('root ::= a a\na ::= "x" | "xx" | ""')
    assert [n for n in range(6) if ambiguous.accepts("x" * n)] == [0, 1, 2, 3, 4]
    with pytest.raises(TypeError):
        Grammar.parse(b"root ::= [a]")


def test_grammar_oracle_judges_a_token_mask_over_the_toy_vocabulary():
    """What `grammar-token-mask` checks: a token is allowed exactly when the bytes so
    far plus the token's bytes are a prefix of the language."""
    vocab = ToyVocab(64, seed=0)
    g = Grammar.parse(
        'root ::= "{" pair ("," " "? pair)* "}"\npair ::= "\\"" [a-z]+ "\\":" " "? value\n'
        'value ::= "true" | "false" | "null" | [0-9]+ | root'
    )
    assert g.accepts('{"a":true, "nul": {"e":12}}')
    so_far = b'{"a": '
    allowed = {
        t for t, piece in enumerate(vocab.pieces) if piece and g.accepts_prefix(so_far + piece)
    }
    texts = {vocab.pieces[t] for t in allowed}
    assert {b"tr", b"true", b"fals", b"nul", b"1", b"{", b'{"'} <= texts
    assert not texts & {b"}", b",", b'"', b"a", b" ", b'":', b"\xe2\x82"}
    # A token that spans two rules (`":` ends the key and starts the separator).
    assert g.accepts_prefix(b'{"a' + b'":') and not g.accepts_prefix(b'{"' + b'":')


# -- ScriptedChatModel and ScriptedTool ---------------------------------------------------


def user(text):
    return {"role": "user", "content": text}


def tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_chat_model_replays_tool_calls_and_final_answers():
    model = ScriptedChatModel(
        [
            reply(tool_calls=[tool_call("get_weather", {"city": "Paris"}),
                              tool_call("get_time", {"zone": "CET"})]),
            [tool_call("get_weather", '{"city": ', id="broken")],
            "21 degrees.",
        ],
        cost=2.5,
    )  # fmt: skip
    messages = [user("weather in Paris?")]
    first = model.complete(messages)
    assert first == {
        "role": "assistant", "content": None,
        "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
            {"id": "call_1", "type": "function",
             "function": {"name": "get_time", "arguments": '{"zone": "CET"}'}},
        ],
    }  # fmt: skip
    first["tool_calls"].clear()  # the caller's copy: the model's record is unaffected
    assert len(model.replies[0]["tool_calls"]) == 2
    messages += [model.replies[0], tool_result("call_1", "12:00"), tool_result("call_0", "21 C")]
    second = model.complete(messages)
    (call,) = second["tool_calls"]
    assert call["id"] == "broken" and call["function"]["arguments"] == '{"city": '
    with pytest.raises(json.JSONDecodeError):
        json.loads(call["function"]["arguments"])  # malformed on purpose
    messages += [second, tool_result("broken", "error: arguments are not valid JSON")]
    assert model.complete(messages) == {"role": "assistant", "content": "21 degrees."}
    assert model.counters.snapshot() == {"calls": 3, "tool_calls": 3}
    assert model.spent == 7.5 and model.violations == []
    assert [len(c) for c in model.calls] == [1, 4, 6] and model.calls[2] == messages
    with pytest.raises(ScriptExhausted, match="called 4 times, but its script has 3"):
        model.complete(messages)
    twin = model.with_framework(None)
    assert (
        twin.counters.calls == 0
        and twin.complete([user("again")])["tool_calls"][0]["id"] == "call_0"
    )


def test_chat_model_validates_the_conversation():
    def fresh():
        model = ScriptedChatModel([[tool_call("a"), tool_call("b")], "done"], repeat_last=True)
        return model, [user("go"), model.complete([user("go")])]

    for extra, code in [
        ([], "missing_tool_result"),  # nothing answered
        ([tool_result("call_0", "x")], "missing_tool_result"),  # only the first of two
        ([tool_result("call_0", "x"), tool_result("call_7", "y")], "unknown_tool_call_id"),
        ([tool_result("call_0", "x"), tool_result("call_0", "y")], "duplicate_tool_result"),
        (
            [tool_result("call_0", "x"), user("hm"), tool_result("call_1", "y")],
            "missing_tool_result",
        ),
        ([{"role": "tool", "content": "x"}], "bad_message"),
        ([{"role": "robot", "content": "x"}], "bad_message"),
    ]:
        model, messages = fresh()
        with pytest.raises(ChatProtocolError) as caught:
            model.complete(messages + extra)
        assert caught.value.code == code and model.violations == [caught.value]
        assert model.counters.calls == 1  # the refused call did not consume the script
    model, messages = fresh()
    done = messages + [tool_result("call_1", "y"), tool_result("call_0", "x")]  # any order
    assert model.complete(done)["content"] == "done"
    assert model.complete(done)["content"] == "done"  # repeat_last
    with pytest.raises(ChatProtocolError, match="list of message dicts"):
        model.complete("hello")
    lenient = ScriptedChatModel(["ok"], strict=False)
    assert lenient.complete([{"role": "tool", "content": "x"}])["content"] == "ok"


def test_chat_model_scripts_by_content_or_function_and_carries_logprobs():
    keyed = ScriptedChatModel({"hi": "hello", None: reply("?", logprobs=[-0.5, -1.5])})
    assert keyed.complete([user("hi")]) == {"role": "assistant", "content": "hello"}
    assert keyed.complete([user("what")])["logprobs"] == [-0.5, -1.5]
    with pytest.raises(KeyError):
        ScriptedChatModel({"hi": "hello"}).complete([user("what")])
    counting = ScriptedChatModel(lambda messages, index: f"call {index}: {len(messages)} messages")
    assert counting.complete([user("a")])["content"] == "call 0: 1 messages"
    assert counting.complete([user("a"), user("b")])["content"] == "call 1: 2 messages"
    assert ScriptedChatModel([{"content": "plain dict"}]).complete([])["content"] == "plain dict"
    with pytest.raises(TypeError, match="a scripted reply is"):
        ScriptedChatModel([42]).complete([])
    with pytest.raises(ValueError, match="built with tool_call"):
        reply(tool_calls=[{"name": "x"}])
    # A cascade: confidence is the mean log-probability, the price is known up front.
    small = ScriptedChatModel([reply("maybe", logprobs=[-2.0, -1.0])], name="small", cost=1.0)
    large = ScriptedChatModel(
        [reply("surely", logprobs=[-0.1, -0.1, -0.1])], name="large", cost=10.0
    )
    answer, budget = None, 12.0
    for model, threshold in ((small, -1.0), (large, -0.5)):
        if model.spent + small.spent * (model is large) + model.cost > budget:
            break
        message = model.complete([user("q")])
        if sum(message["logprobs"]) / len(message["logprobs"]) >= threshold:
            answer = message["content"]
            break
    assert answer == "surely" and (small.counters.calls, large.counters.calls) == (1, 1)
    assert small.spent + large.spent == 11.0


def test_chat_model_streams_the_same_reply_in_pieces_on_the_virtual_clock(sim):
    model = ScriptedChatModel(
        [reply("Let me check.", tool_calls=[tool_call("search", {"q": "kv cache"})]), "Done."],
        delay_s=0.25, chunk_chars=6,
    )  # fmt: skip
    seen = []

    async def consume():
        async for delta in model.stream([user("go")]):
            seen.append((sim.now(), delta))
        message = await model.acomplete(
            [user("go"), model.replies[0], tool_result("call_0", "found")]
        )
        seen.append((sim.now(), message))

    sim.spawn(consume())
    sim.run()
    deltas = [d for _, d in seen[:-1]]
    assert [d.get("content") for d in deltas[:3]] == ["Let me", " check", "."]
    assert deltas[3] == {"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                                         "function": {"name": "search", "arguments": ""}}]}  # fmt: skip
    fragments = [d["tool_calls"][0]["function"]["arguments"] for d in deltas[4:-1]]
    assert "".join(fragments) == '{"q": "kv cache"}' and all(len(f) <= 6 for f in fragments)
    assert deltas[-1] == {"finish_reason": "tool_calls"}
    assert [t for t, _ in seen[:-1]] == [0.25 * (i + 1) for i in range(len(deltas))]
    assert assemble(deltas) == model.replies[0]
    assert seen[-1] == (0.25 * (len(deltas) + 1), {"role": "assistant", "content": "Done."})
    assert assemble([{"content": "a"}, {"content": "b"}, {"finish_reason": "stop"}]) == {
        "role": "assistant", "content": "ab",
    }  # fmt: skip


def test_scripted_tool_records_runs_failures_and_cancellations(sim):
    def lookup(city):
        if city == "Atlantis":
            raise KeyError(city)
        return f"21 C in {city}"

    weather = ScriptedTool("get_weather", lookup, delay_s=0.5, description="Current weather.",
                           parameters={"type": "object", "properties": {"city": {"type": "string"}}})  # fmt: skip
    charge = ScriptedTool("charge_card", lambda amount: "ok", side_effects=True)
    assert weather.schema == {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}  # fmt: skip
    assert charge.side_effects and not weather.side_effects
    results = {}

    async def scenario():
        speculative = asyncio.ensure_future(weather(city="Paris"))
        both = asyncio.gather(
            weather(city="Oslo"), weather(city="Atlantis"), return_exceptions=True
        )
        await asyncio.sleep(0.25)
        speculative.cancel()  # a misprediction: the tool function never runs for it
        results["both"] = await both
        results["at"] = sim.now()

    sim.spawn(scenario())
    sim.run()
    assert results["both"][0] == "21 C in Oslo" and isinstance(results["both"][1], KeyError)
    assert results["at"] == 0.5  # concurrent: half a second, not one
    assert [(r.arguments, r.status, r.started_s, r.finished_s) for r in weather.runs] == [
        ({"city": "Paris"}, "canceled", 0.0, 0.25),
        ({"city": "Oslo"}, "done", 0.0, 0.5),
        ({"city": "Atlantis"}, "failed", 0.0, 0.5),
    ]
    assert weather.runs[0].result is None and weather.runs[1].result == "21 C in Oslo"
    assert weather.counters.snapshot() == {"started": 3, "done": 1, "failed": 1, "canceled": 1}
    assert charge.run(amount=5) == "ok" and charge.runs[0].status == "done"


# -- vectors ------------------------------------------------------------------------------


def test_vector_sets_have_known_neighbors():
    rng = np.random.default_rng(0)
    data = vectors.clustered(rng, n=400, dim=16, clusters=5)
    assert data.vectors.shape == (400, 16) and data.vectors.dtype == np.float32
    assert data.centroids.shape == (5, 16) and data.labels.shape == (400,)
    np.testing.assert_allclose(np.linalg.norm(data.vectors, axis=1), 1.0, rtol=1e-5)
    # Most vectors are nearest to the centroid they were drawn around.
    nearest, _ = vectors.exact_topk(data.vectors, data.centroids, 1)
    assert (nearest[:, 0] == data.labels).mean() > 0.95
    planted = [3, 17, 256, 399]
    queries = vectors.queries_near(rng, data.vectors, planted, noise=0.05)
    assert queries.shape == (4, 16) and queries.dtype == np.float32
    for metric in vectors.METRICS:
        found, scores = vectors.exact_topk(queries, data.vectors, 10, metric=metric)
        assert found.shape == scores.shape == (4, 10) and found.dtype == np.int64
        assert found[:, 0].tolist() == planted and (np.diff(scores, axis=1) <= 0).all()
    # Ties go to the lower index; k beyond the corpus is clipped.
    twice = np.concatenate([data.vectors[:3], data.vectors[:3]])
    found, _ = vectors.exact_topk(data.vectors[1], twice, 99)
    assert found.shape == (1, 6) and found[0, :2].tolist() == [1, 4]
    assert vectors.recall([[1, 2, 3], [4, 5, 6]], [[3, 2, 9], [4, 5, 6]]) == pytest.approx(5 / 6)
    again = vectors.clustered(np.random.default_rng(0), n=400, dim=16, clusters=5)
    np.testing.assert_array_equal(again.vectors, data.vectors)
    with pytest.raises(ValueError, match="metric must be one of"):
        vectors.exact_topk(queries, data.vectors, 1, metric="manhattan")


def test_counting_distance_counts_the_vectors_it_scans(fw):
    data = vectors.clustered(np.random.default_rng(1), n=60, dim=8, clusters=4)
    distance = vectors.CountingDistance(fw, metric="l2")
    query = data.vectors[7]
    d = distance(fw.tensor(query), fw.tensor(data.vectors[:20]))
    assert fw.is_tensor(d) and fw.numpy(d).shape == (20,) and fw.numpy(d).dtype == np.float32
    np.testing.assert_allclose(fw.numpy(d), ((data.vectors[:20] - query) ** 2).sum(-1), atol=1e-5)
    table = distance(data.vectors[:3], data.vectors)
    assert fw.numpy(table).shape == (3, 60)
    assert fw.numpy(distance(query, data.vectors[:0])).shape == (0,)  # an empty list is fine
    assert distance.counters.snapshot() == {"calls": 3, "distances": 20 + 180}

    # What `ivf-index-search` checks: distances == n_centroids + sizes of probed lists.
    ivf = distance.with_framework(fw)
    lists = np.asarray(fw.numpy(ivf(data.vectors, data.centroids))).argmin(axis=1)
    ivf.counters.reset()
    order = np.argsort(np.asarray(fw.numpy(ivf(query, data.centroids))), kind="stable")
    probed = [np.nonzero(lists == c)[0] for c in order[:2]]
    candidates = np.concatenate(probed)
    scores = np.asarray(fw.numpy(ivf(query, data.vectors[candidates])))
    assert candidates[scores.argmin()] == 7
    assert ivf.counters.distances == 4 + sum(len(p) for p in probed) < 60
    dot = vectors.CountingDistance(fw, metric="dot")
    np.testing.assert_allclose(
        fw.numpy(dot(query, data.vectors[:5])), data.vectors[:5] @ query, atol=1e-5
    )
    with pytest.raises(ValueError, match="expected queries"):
        distance(query, query)


# -- workloads ----------------------------------------------------------------------------


def test_phased_workload_follows_the_load_steps():
    requests = workloads.phases(
        np.random.default_rng(0), [(2.0, 5.0), (4.0, 40.0), (2.0, 0.0), (1.0, 10.0)],
        quantum_s=2**-10, tenants=("a", "b"), start_s=1.0,
    )  # fmt: skip
    times = [r.arrival_s for r in requests]
    assert times == sorted(times) and [r.id for r in requests] == list(range(len(requests)))
    count = lambda lo, hi: sum(1 for t in times if lo <= t < hi)
    assert 3 <= count(1, 3) <= 20 and 120 <= count(3, 7) <= 200
    assert count(7, 9) == 0 and count(9, 10) >= 1 and count(10, 99) == 0 and count(0, 1) == 0
    assert {r.tenant for r in requests} == {"a", "b"} and all(r.tokens is None for r in requests)
    again = workloads.phases(np.random.default_rng(0), [(2.0, 5.0), (4.0, 40.0), (2.0, 0.0),
                                                        (1.0, 10.0)], quantum_s=2**-10, start_s=1.0)  # fmt: skip
    assert [r.arrival_s for r in again] == times  # adding tenants= moved no arrival
    assert workloads.phases(np.random.default_rng(0), []) == []
    with pytest.raises(ValueError, match="a phase is"):
        workloads.phases(np.random.default_rng(0), [(1.0, -1.0)])


def test_shared_prefix_workload_carries_token_ids():
    rng = np.random.default_rng(3)
    sequences = workloads.token_sequences(rng, 40, groups=3, prefix_tokens=20, suffix_tokens=(2, 6))
    heads = {s[:20] for s in sequences}
    assert len(heads) == 3 and all(22 <= len(s) <= 26 for s in sequences)
    skewed = workloads.token_sequences(np.random.default_rng(3), 300, groups=4, prefix_tokens=8,
                                       suffix_tokens=1, skew=2.0)  # fmt: skip
    counts = sorted(
        (sum(1 for s in skewed if s[:8] == h) for h in {s[:8] for s in skewed}), reverse=True
    )
    assert counts[0] > 150 > counts[-1]  # a hot prefix

    requests = workloads.shared_prefixes(np.random.default_rng(4), rate=10.0, n=30, groups=2,
                                         prefix_tokens=(16, 32), output_tokens=3)  # fmt: skip
    assert all(r.tokens is not None and r.prompt_tokens == len(r.tokens) for r in requests)
    assert (
        all(r.output_tokens == 3 for r in requests) and len({r.tokens[:16] for r in requests}) == 2
    )
    assert [r.arrival_s for r in requests] == sorted(r.arrival_s for r in requests)
    with pytest.raises(ValueError, match="tokens has 2 ids but prompt_tokens is 3"):
        SimRequest(0, 0.0, 3, 1, tokens=(1, 2))
    with pytest.raises(ValueError, match="sequence of ints"):
        SimRequest(0, 0.0, 2, 1, tokens=(1, "2"))
    assert SimRequest(0, 0.0, 2, 1, tokens=[7, 8]).tokens == (7, 8)
    assert workloads.replay([(0.0, 2, 1, 0, None, None, (5, 6))])[0].tokens == (5, 6)


# -- SimEngine: sequences that do not start from nothing ----------------------------------


def test_engine_admits_a_cached_prefix_and_a_resumed_sequence(sim):
    engine = SimEngine(sim, COST, kv_blocks=6, block_tokens=4, name="e0")
    engine.admit(req("hit", 10, 2), cached_tokens=8)
    assert engine.remaining_prefill("hit") == 2 and engine.kv_tokens("hit") == 8
    assert engine.blocks("hit") == 2 and engine.free_blocks == 4
    engine.admit(req("moved", 5, 3), resume_from=1)
    assert engine.remaining_prefill("moved") == 0 and engine.generated("moved") == 1
    assert engine.kv_tokens("moved") == 5 and engine.free_blocks == 2

    async def run():
        first = await engine.step(prefill=[("hit", 2)], decode=["moved"])
        second = await engine.step(decode=["hit", "moved"])
        return first, second

    first, second = sim.run_until_complete(run())
    assert first.duration == 0.5 + 2 * 0.125 + 0.25 and first.tokens == ("hit", "moved")
    assert second.finished == ("hit", "moved") and engine.live == []
    kinds = [(e.kind, e.fields) for e in sim.trace.events]
    assert kinds[0] == ("admit", {"id": "hit", "cached": 8, "engine": "e0"})
    assert kinds[1] == ("admit", {"id": "moved", "resumed": 1, "engine": "e0"})
    # The resumed sequence already had its first token elsewhere: no second one here.
    assert [e["id"] for e in sim.trace.of("first_token")] == ["hit"]
    assert [(e["id"], e["index"]) for e in sim.trace.of("token")] == [
        ("hit", 0), ("moved", 1), ("hit", 1), ("moved", 2),
    ]  # fmt: skip
    assert all(e["engine"] == "e0" for e in sim.trace.events)
    assert engine.counters.prefill_tokens == 2 and engine.counters.output_tokens == 4

    for kwargs, code in [
        ({"cached_tokens": 10}, "bad_request"),  # at least one token must be prefilled
        ({"cached_tokens": -1}, "bad_request"),
        ({"resume_from": 2}, "bad_request"),  # nothing would be left to decode
        ({"cached_tokens": 4, "resume_from": 1}, "bad_request"),
    ]:
        with pytest.raises(Exception, match=".") as caught:
            engine.admit(req("x", 10, 2), **kwargs)
        assert caught.value.code == code and engine.live == []
    engine.admit(req("big", 12, 2), cached_tokens=8)
    with pytest.raises(Exception, match="blocks are free") as caught:
        engine.admit(req("more", 24, 2), cached_tokens=20)
    assert caught.value.code == "out_of_blocks" and engine.live == ["big"]
    plain = SimEngine(sim, COST)
    plain.admit(req("p", 3, 1))
    assert sim.trace.events[-1].fields == {"id": "p"}  # unnamed, unchanged


# -- SimCluster ---------------------------------------------------------------------------


def served(cluster):
    assert cluster.violations == [], [str(v) for v in cluster.violations]
    assert all(r.engine.violations == [] for r in cluster.all_replicas)
    return cluster.outcomes


def test_cluster_replica_runs_continuous_batching_first_come_first_served(sim):
    cluster = SimCluster(sim, COST, replicas=1, max_seqs=2, max_batch_tokens=8)
    (replica,) = cluster.replicas
    requests = [req("a", 6, 2), req("b", 5, 3), req("c", 2, 1, arrival=0.25)]
    workloads.drive(sim, requests, cluster.submit)
    sim.run()
    steps = [(e.time, e["prefill"], e["decode"]) for e in sim.trace.of("step")]
    assert steps == [
        (0.0, (("a", 6), ("b", 2)), ()),  # FCFS prefill inside the 8-token budget
        (1.5, (("b", 3),), ("a",)),  # a decodes; the rest of b's prompt
        (2.625, (("c", 2),), ("b",)),  # a finished: c is admitted (max_seqs=2)
        (3.625, (), ("b",)),
    ]
    out = served(cluster)
    assert (out["a"].first_token_s, out["a"].finish_s) == (1.5, 2.625)
    assert (out["b"].first_token_s, out["b"].finish_s) == (2.625, 4.375)
    assert (out["c"].submitted_s, out["c"].first_token_s, out["c"].finish_s) == (0.25, 3.625, 3.625)
    assert all(
        o.status == "finished" and o.replica == 0 and o.phase == "full" for o in out.values()
    )
    assert all(e["engine"] == 0 for e in sim.trace.of("step"))
    assert sim.trace.latencies("arrive", "first_token") == [1.5, 2.625, 3.375]
    assert cluster.counters.routed == cluster.counters.finished == 3
    assert (replica.queue_len, replica.outstanding_tokens, replica.kv_usage) == (0, 0, 0.0)
    assert cluster.replica_seconds() == 4.375


def test_cluster_routes_by_hook_and_reports_load(sim):
    seen = []

    def route(request, replicas):
        seen.append([(r.id, r.queue_len, r.waiting_len, r.running_len, r.outstanding_tokens)
                     for r in replicas])  # fmt: skip
        return replicas[-1] if request.id == "b" else 0  # a replica or its id

    cluster = SimCluster(sim, COST, replicas=2, max_seqs=1, route=route)
    futures = [cluster.submit(req("a", 4, 2)), cluster.submit(req("b", 3, 1)),
               cluster.submit(req("c", 2, 2))]  # fmt: skip
    assert seen == [
        [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0)],
        [(0, 1, 1, 0, 6), (1, 0, 0, 0, 0)],
        [(0, 1, 1, 0, 6), (1, 1, 1, 0, 4)],
    ]
    sim.run(until=0.25)
    zero = cluster.replica(0)
    assert (zero.running_len, zero.waiting_len, zero.outstanding_tokens) == (1, 1, 6 + 4)
    sim.run()
    assert [f.result().replica for f in futures] == [0, 1, 0]
    assert [(e["id"], e["replica"]) for e in sim.trace.of("route")] == [
        ("a", 0),
        ("b", 1),
        ("c", 0),
    ]
    assert served(cluster)["c"].first_token_s > served(cluster)["a"].finish_s
    # Without a hook: the shortest queue, the lowest id on ties.
    default = SimCluster(sim, COST, replicas=3)
    picks = [default.submit(req(i, 2, 1)) for i in range(4)]
    sim.run()
    assert [f.result().replica for f in picks] == [0, 1, 2, 0]


def test_cluster_prefix_cache_skips_what_a_replica_has_prefilled(sim):
    cluster = SimCluster(sim, COST, replicas=2, cache_tokens=64, block_tokens=4,
                         route=lambda request, replicas: request.priority)  # fmt: skip
    shared = tuple(range(1, 11))  # ten tokens in common
    first = SimRequest("first", 0.0, 12, 1, tokens=(*shared, 50, 51))
    again = SimRequest("again", 4.0, 13, 1, tokens=(*shared, 60, 61, 62))
    elsewhere = SimRequest("elsewhere", 4.0, 13, 1, priority=1, tokens=(*shared, 60, 61, 62))
    same = SimRequest("same", 8.0, 12, 1, tokens=(*shared, 50, 51))
    plain = SimRequest("plain", 8.0, 12, 1, priority=1)  # no token ids: never a hit
    workloads.drive(sim, [first, again, elsewhere, same, plain], cluster.submit)
    sim.run()
    out = served(cluster)
    # 10 shared tokens are 2 whole blocks of 4; an identical prompt keeps one token back.
    assert {k: o.cached_tokens for k, o in out.items()} == {
        "first": 0, "again": 8, "elsewhere": 0, "same": 8, "plain": 0,
    }  # fmt: skip
    assert [(e["id"], e["replica"], e["hit"]) for e in sim.trace.of("cache_lookup")] == [
        ("first", 0, 0),
        ("again", 0, 8),
        ("elsewhere", 1, 0),
        ("same", 0, 8),
    ]
    assert cluster.counters.cache_lookups == 4 and cluster.counters.cache_hit_tokens == 16
    assert out["again"].first_token_s - 4.0 == 0.5 + 5 * 0.125  # 13 - 8 tokens prefilled
    assert out["elsewhere"].first_token_s - 4.0 == 0.5 + 13 * 0.125
    cache = cluster.replica(0).cache
    assert cache.match(again.tokens) == 12 and cache.match((9, 9, 9)) == 0
    assert cache.cached_tokens == 12 + 13  # `same` was already there
    for i in range(6):  # 64 tokens of capacity: the least recently used prompts go
        cache.insert(tuple(range(100 * i, 100 * i + 12)))
    assert cache.cached_tokens <= 64 and cache.match(first.tokens) == 0


def test_cluster_prefix_aware_routing_beats_round_robin_on_cache_hits(sim):
    """The comparison `prefix-aware-router` makes, with the routers written here."""
    requests = workloads.shared_prefixes(np.random.default_rng(5), rate=4.0, n=60, groups=3,
                                         prefix_tokens=(40, 48), suffix_tokens=(2, 6),
                                         output_tokens=(2, 4), quantum_s=2**-10)  # fmt: skip
    cost = CostModel(base_s=2**-6, prefill_s_per_token=2**-8, decode_s_per_seq=2**-7)

    def run(make_route):
        with SimClock() as clock:
            # Room for one prompt per replica: a replica that sees every prefix in turn
            # keeps evicting the one it needs next.
            cluster = SimCluster(clock, cost, replicas=3, cache_tokens=64, block_tokens=8,
                                 route=make_route())  # fmt: skip
            workloads.drive(clock, requests, cluster.submit)
            clock.run()
            assert len(served(cluster)) == 60
            ttft = clock.trace.latencies("arrive", "first_token")
            return cluster.counters.cache_hit_tokens, percentile(ttft, 95)

    def round_robin():
        turn = itertools.count()
        return lambda request, replicas: replicas[next(turn) % len(replicas)]

    def by_prefix():
        index = {}  # the router's own record of where each prefix went

        def route(request, replicas):
            home = index.setdefault(request.tokens[:40], len(index) % len(replicas))
            return replicas[home]

        return route

    rr_hits, rr_p95 = run(round_robin)
    aware_hits, aware_p95 = run(by_prefix)
    assert aware_hits > 1.5 * rr_hits and aware_hits >= 40 * 57
    assert aware_p95 < rr_p95


def test_cluster_scales_up_with_a_startup_delay_and_drains_without_failures(sim):
    cluster = SimCluster(sim, COST, replicas=1, max_seqs=1, startup_s=2.0)
    for i in range(4):
        cluster.submit(req(i, 4, 1))  # a queue on the only replica
    metrics = cluster.metrics()
    assert (metrics.ready, metrics.starting, metrics.queue_depth, metrics.running) == (1, 0, 4, 0)
    sim.run(until=0.5)
    cluster.scale_to(3)  # two more, ready at 2.5
    metrics = cluster.metrics()
    assert (metrics.ready, metrics.starting, metrics.draining) == (1, 2, 0)
    assert (metrics.queue_depth, metrics.running, metrics.outstanding_tokens) == (3, 1, 20)
    assert [r.state for r in cluster.replicas] == ["ready", "starting", "starting"]
    late = cluster.submit(req("late", 4, 1))  # goes to a ready replica while one exists
    sim.run(until=2.5)
    assert [r.state for r in cluster.replicas] == ["ready"] * 3
    quick = cluster.submit(req("quick", 4, 1))
    sim.run(until=2.75)
    assert quick.result().replica == 1 if quick.done() else cluster.replica(1).queue_len == 1
    cluster.scale_to(1)  # the newest go first, and they finish what they hold
    assert [r.state for r in cluster.all_replicas] == ["ready", "draining", "draining"]
    with pytest.raises(ClusterError, match="replica 1, which is draining") as caught:
        cluster.replica(1).submit(req("no", 1, 1))
    assert caught.value.code == "replica_unavailable" and cluster.violations == [caught.value]
    cluster.violations.clear()
    sim.run()
    out = served(cluster)
    assert all(o.status == "finished" for o in out.values()) and len(out) == 6
    assert out["quick"].replica == 1 and late.result().replica == 0
    assert [r.state for r in cluster.all_replicas] == ["ready", "stopped", "stopped"]
    assert [r.id for r in cluster.replicas] == [0]
    assert cluster.counters.scale_ups == 2 and cluster.counters.scale_downs == 2
    stops = {e["replica"]: e.time for e in sim.trace.of("replica_stop")}
    assert stops[2] == 2.75 and stops[1] == out["quick"].finish_s  # idle: at once; busy: when done
    assert [e.time for e in sim.trace.of("replica_ready")] == [0.0, 2.5, 2.5]
    assert cluster.replica_seconds() == sim.now() + (stops[1] - 0.5) + (stops[2] - 0.5)


def test_cluster_stop_fails_what_a_replica_holds(sim):
    cluster = SimCluster(sim, COST, replicas=2, max_seqs=1)
    futures = [cluster.replica(1).submit(req(i, 4, 3)) for i in range(2)]
    safe = cluster.replica(0).submit(req("safe", 4, 1))
    sim.run(until=1.25)  # request 0 has its first token; request 1 is waiting
    cluster.stop(1)
    cluster.stop(1)  # idempotent
    sim.run()
    assert [(f.result().status, f.result().reason) for f in futures] == [
        ("failed", "replica_stopped")
    ] * 2
    assert futures[0].result().first_token_s == 1.0 and futures[0].result().finish_s is None
    assert safe.result().status == "finished"
    assert [(e["id"], e["reason"]) for e in sim.trace.of("fail")] == [
        (0, "replica_stopped"),
        (1, "replica_stopped"),
    ]
    assert (cluster.counters.failed, cluster.counters.finished, cluster.counters.scale_downs) == (
        2,
        1,
        1,
    )
    assert cluster.replica(1).engine.live == [] and cluster.replica(1).engine.violations == []
    cluster.stop(0)
    nowhere = cluster.submit(req("nowhere", 1, 1))
    assert nowhere.result().reason == "no_replica" and nowhere.result().replica is None
    with pytest.raises(ValueError, match="there is no replica 5"):
        cluster.replica(5)
    with pytest.raises(TypeError):
        cluster.replica("0")
    with pytest.raises(ValueError, match="scale_to"):
        cluster.scale_to(-1)


def test_cluster_kv_pool_limits_admission_and_reports_usage(sim):
    cluster = SimCluster(sim, COST, replicas=1, max_seqs=4, kv_blocks=4, block_tokens=4)
    (replica,) = cluster.replicas
    a = cluster.submit(req("a", 7, 2))  # worst case 8 tokens: 2 blocks
    b = cluster.submit(req("b", 8, 1))  # 2 blocks
    c = cluster.submit(req("c", 3, 1))  # must wait for memory, although max_seqs allows it
    huge = cluster.submit(req("huge", 20, 1))  # can never fit: fails instead of blocking
    assert huge.result().status == "failed" and huge.result().reason == "too_large"
    sim.run(until=0.25)
    assert (replica.running_len, replica.waiting_len, replica.kv_usage) == (2, 1, 1.0)
    assert cluster.metrics().kv_usage == 1.0
    sim.run()
    assert all(f.result().status == "finished" for f in (a, b, c))
    assert c.result().first_token_s > b.result().finish_s and replica.kv_usage == 0.0
    served(cluster)


def test_cluster_prefill_on_one_replica_and_decode_on_another(sim):
    """What `prefill-decode-disaggregation` needs: a prefill phase, a metered
    transfer, a decode phase, and decode steps that never carry a prefill."""
    cluster = SimCluster(sim, COST, replicas=2, transfer_s_per_token=0.0625)
    prefiller, decoder = cluster.replicas
    request = req("r", 8, 3)

    async def serve():
        first = await prefiller.submit(request, phase="prefill")
        await cluster.transfer(request, prefiller, decoder)
        last = await decoder.submit(request, phase="decode")
        return first, last

    local = decoder.submit(req("local", 2, 4))  # a short prompt stays on the decode worker
    first, last = sim.run_until_complete(serve())
    sim.run()
    assert (first.phase, first.first_token_s, first.finish_s) == ("prefill", 1.5, 1.5)
    assert (last.phase, last.first_token_s, last.replica, last.status) == (
        "decode",
        None,
        1,
        "finished",
    )
    transfer = sim.trace.of("transfer")[0]
    assert (transfer.time, transfer["tokens"], transfer["duration"]) == (1.5, 8, 0.5)
    assert sim.trace.of("transfer_done")[0].time == 2.0
    assert cluster.counters.transfers == 1 and cluster.counters.transferred_tokens == 8
    # Exactly one first token, and tokens 0, 1, 2 in order across the two replicas.
    assert [(e["id"], e["engine"]) for e in sim.trace.of("first_token")] == [("local", 1), ("r", 0)]
    assert [(e["index"], e["engine"]) for e in sim.trace.of("token") if e["id"] == "r"] == [
        (0, 0), (1, 1), (2, 1),
    ]  # fmt: skip
    assert [e["id"] for e in sim.trace.of("finish")] == ["local", "r"]
    decode_steps = [e for e in sim.trace.of("step") if e["engine"] == 1]
    assert [e["prefill"] for e in decode_steps[1:]] == [()] * (len(decode_steps) - 1)
    assert prefiller.engine.live == [] and served(cluster)["r"] is last
    assert local.result().status == "finished"
    with pytest.raises(ClusterError) as caught:
        decoder.submit(req("one", 4, 1), phase="decode")
    assert caught.value.code == "nothing_to_decode"
    with pytest.raises(ValueError, match="phase must be one of"):
        decoder.submit(request, phase="warmup")
