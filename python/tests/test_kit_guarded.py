"""The kit's Guarded mixin and DeltaMemory: what a solution can and cannot reach."""

import copy
import textwrap
import types

import numpy as np
import pytest

from learnai.judge import frameworks
from learnai.judge.errors import CheckFailure
from learnai.kit import (
    BlockDenoiser,
    Counters,
    CountingExperts,
    CountingKV,
    DeltaMemory,
    Guarded,
    TinyDSA,
    TinyLM,
    check_guard,
    guard,
)
from learnai.kit import guarded as guarded_module


class Meter:
    """A minimal kit object: a private array, a counter, a property, a test-only method."""

    def __init__(self, values):
        self._values = list(values)
        self.size = len(self._values)
        self.counters = Counters("reads")

    @property
    def last_index(self):
        return len(self._values) - 1

    def read(self, i):
        self.counters.add("reads")
        return self._values[i]

    def grown(self, value):
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._values = [*self._values, value]
        return new

    def peek(self):
        return list(self._values)

    def __repr__(self):
        return f"Meter({self.counters!r})"


class GuardedMeter(Guarded, Meter):
    label = "meter"
    refused = Guarded.refused | {"peek"}


class ListedMeter(Guarded, Meter):
    label = "meter"
    allowed = frozenset({"read", "size"})


@pytest.fixture
def solve(tmp_path):
    """Compile source as the solution file, arm the guard, and return its namespace."""

    def make(source):
        path = tmp_path / "solution.py"
        path.write_text(textwrap.dedent(source))
        namespace = {}
        exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102
        guard(types.SimpleNamespace(solution=types.SimpleNamespace(__file__=str(path))))
        return namespace

    yield make
    guarded_module._solution_file = None
    guarded_module._violations.clear()


def refused(function, *args):
    """Call a solution function that must be refused; return the recorded message."""
    guarded_module._violations.clear()
    with pytest.raises(CheckFailure) as info:
        function(*args)
    with pytest.raises(CheckFailure) as recorded:
        check_guard()
    assert recorded.value.message == info.value.message
    return info.value.message


# -- what the solution may do --------------------------------------------------------


def test_the_solution_uses_public_methods_sizes_and_properties(solve):
    solution = solve(
        """
        def use(meter):
            counters = 0                      # a learner's own name: nobody's business
            longer = meter.grown(7)
            return meter.read(1) + longer.read(3) + meter.size + meter.last_index + counters
        """
    )
    meter = GuardedMeter([1, 2, 3])
    assert solution["use"](meter) == 2 + 7 + 3 + 2
    check_guard()
    assert meter.counters.reads == 2  # the grown copy is guarded and shares the counters


def test_printing_a_guarded_object_is_allowed(solve):
    solution = solve("def show(meter): return repr(meter) + str(meter) + f'{meter}'")
    assert solution["show"](GuardedMeter([1])) == "Meter(Counters(reads=0))" * 3
    check_guard()


def test_kit_and_test_code_is_not_restricted(solve):
    solve("def nothing(): pass")
    meter = GuardedMeter([1, 2])
    meter.counters.reset()
    meter.note = "set by the test"
    assert meter._values == [1, 2] and meter.peek() == [1, 2] and meter.note
    assert vars(meter)["size"] == 2 and "counters" in meter.__dict__
    del meter.note
    assert not hasattr(meter, "note")
    assert Meter.peek(meter) == [1, 2]  # also as a plain function, outside the solution
    check_guard()


def test_nothing_is_refused_before_a_guard_is_armed():
    assert guarded_module._solution_file is None
    meter = GuardedMeter([1])
    assert meter.counters.reads == 0 and meter._values == [1]


# -- what it may not -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("meter.counters.reset()", "`meter.counters` may not be used in this problem"),
        ("meter._values", "`meter._values` may not be used in this problem"),
        ("meter.peek()", "`meter.peek` may not be used in this problem"),
        ("meter.with_framework", "`meter.with_framework` may not be used in this problem"),
        ("vars(meter)", "`meter.__dict__` may not be used in this problem"),
        ("getattr(meter, 'coun' + 'ters')", "`meter.counters` may not be used in this problem"),
        ("eval('m.counters', {'m': meter})", "`meter.counters` may not be used in this problem"),
        (
            "import operator; operator.attrgetter('counters')(meter)",
            "`meter.counters` may not be used in this problem",
        ),
        ("meter.size = 0", "`meter.size` may not be set in this problem"),
        ("meter.stash = [1]", "`meter.stash` may not be set in this problem"),
        ("setattr(meter, 'counters', None)", "`meter.counters` may not be set in this problem"),
        ("del meter.size", "`meter.size` may not be set in this problem"),
        ("import copy; copy.copy(meter)", "`meter` may not be copied in this problem"),
        ("import copy; copy.deepcopy(meter)", "`meter` may not be copied in this problem"),
        ("import pickle; pickle.dumps(meter)", "`meter` may not be copied in this problem"),
    ],
)
def test_the_solution_is_refused(solve, body, message):
    solution = solve(f"def cheat(meter):\n    {body}\n")
    assert refused(solution["cheat"], GuardedMeter([1, 2, 3])) == message


def test_a_swallowed_refusal_still_fails_the_case(solve):
    solution = solve(
        """
        def cheat(meter):
            try:
                meter.counters.reset()
            except Exception:
                pass
            return meter.read(0)
        """
    )
    assert solution["cheat"](GuardedMeter([5])) == 5
    with pytest.raises(CheckFailure, match="`meter.counters` may not be used"):
        check_guard()


def test_arming_again_forgets_earlier_violations(solve):
    solution = solve("def cheat(meter): meter.counters")
    refused(solution["cheat"], GuardedMeter([1]))
    solve("def fine(meter): return meter.read(0)")
    check_guard()


def test_a_borrowed_method_cannot_read_what_the_solution_cannot(solve):
    # `type(meter).__mro__[2]` is Meter: its functions run in kit code, which is trusted,
    # so the restricted names are readable only inside a call made through the object.
    # The same rule stops code compiled under the name of a trusted file.
    solution = solve(
        f"""
        def unbound(meter):
            return type(meter).__mro__[2].peek(meter)

        def spoofed(meter):
            exec(compile("meter._values", {guarded_module.__file__!r}, "exec"))
        """
    )
    meter = GuardedMeter([1, 2])
    message = "`meter._values` may not be read outside the meter's own methods"
    assert refused(solution["unbound"], meter) == message
    with pytest.raises(CheckFailure) as info:
        solution["spoofed"](meter)
    assert info.value.message == message
    assert meter.peek() == [1, 2]  # outside the solution, the test still reads it


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("meter.__getattribute__('counters').reset()", "`meter.__getattribute__` may not be used"),
        ("meter.__setattr__('counters', None)", "`meter.__setattr__` may not be used"),
        ("meter.__delattr__('size')", "`meter.__delattr__` may not be used"),
        ("meter.__init__([0])", "`meter.__init__` may not be used"),
        ("type(meter).__getattribute__(meter, 'counters')", "`meter.counters` may not be used"),
        ("type(meter).__setattr__(meter, 'size', 0)", "`meter.size` may not be set"),
        ("type(meter).__init__(meter, [0])", "`meter._values` may not be set outside the meter's"),
        (
            "type(meter).__mro__[2].__init__(meter, [0])",
            "`meter._values` may not be set outside the meter's own methods",
        ),
        ("type(meter).of(meter)", "`meter.__dict__` may not be read outside the meter's"),
    ],
)
def test_the_machinery_of_the_object_is_not_the_solutions(solve, body, message):
    # Each of these went around the rules: the bound special methods of the guard ran
    # as kit code, and `__init__` starts the counters over.
    solution = solve(f"def cheat(meter):\n    {body}\n")
    meter = GuardedMeter([1, 2, 3])
    meter.read(0)
    assert refused(solution["cheat"], meter).startswith(message)
    assert meter.counters.reads == 1 and meter.size == 3 and meter.peek() == [1, 2, 3]


def test_a_class_that_replaces_refused_keeps_the_machinery_refused(solve):
    class Bare(Guarded, Meter):
        label = "meter"
        refused = frozenset({"peek"})

    solution = solve(
        """
        def state(meter): return vars(meter)
        def again(meter): meter.__init__([0])
        def counted(meter): return meter.counters.reads
        """
    )
    meter = Bare([1])
    assert refused(solution["state"], meter) == "`meter.__dict__` may not be used in this problem"
    assert refused(solution["again"], meter) == "`meter.__init__` may not be used in this problem"
    guarded_module._violations.clear()
    assert solution["counted"](meter) == 0  # this class chose not to refuse its counters
    check_guard()


def test_code_the_solution_has_a_method_run_is_not_a_call_through_the_object(solve):
    # `meter.read(i)` asks for `i.__index__()`. The solution's object answers by borrowing
    # a kit method, inside the kit's own call: still the solution's doing.
    solution = solve(
        """
        def cheat(meter):
            taken = []

            class Index:
                def __index__(self):
                    taken.append(type(meter).__mro__[2].peek(meter))
                    return 0

            return meter.read(Index()), taken
        """
    )
    meter = GuardedMeter([1, 2, 3])
    assert refused(solution["cheat"], meter) == (
        "`meter._values` may not be read outside the meter's own methods"
    )


def test_a_test_callback_may_build_a_guarded_object_through_the_object(solve):
    # A factory of the test that the solution calls directly counts as borrowed; one
    # that is a method of a guarded object is a call through it.
    class Maker(Guarded, Meter):
        label = "maker"

        def make(self, values):
            return GuardedMeter(values)

    def loose(values):
        return GuardedMeter(values)

    solution = solve("def build(factory): return factory([4, 5]).read(1)")
    assert solution["build"](Maker([0]).make) == 5
    check_guard()
    assert refused(solution["build"], loose).startswith("`meter._values` may not be set outside")


# -- introspection --------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "object.__getattribute__(meter, 'counters').reset()",
        "object.__getattribute__(meter, '_values')",
        "object.__getattribute__(meter, '__dict__')['counters'].reset()",
        "type(meter).__mro__[2].__getattribute__(meter, 'counters')",
        "super(type(meter).__mro__[1], meter).__getattribute__('counters')",
    ],
)
def test_the_attributes_are_not_in_the_instance(solve, body):
    # The reviewers' cheat: go around the class's __getattribute__. There is nothing
    # behind it, because a guarded object's attributes are kept off the instance.
    solution = solve(f"def cheat(meter):\n    {body}\n")
    meter = GuardedMeter([1, 2, 3])
    meter.read(0)
    with pytest.raises((AttributeError, KeyError)):
        solution["cheat"](meter)
    assert meter.counters.reads == 1
    assert object.__getattribute__(meter, "__dict__") == {}


def test_writing_around_the_guard_changes_nothing(solve):
    solution = solve(
        """
        def cheat(meter):
            object.__setattr__(meter, "counters", None)
            object.__getattribute__(meter, "__dict__")["_values"] = [9, 9, 9]
            return meter.read(0)
        """
    )
    meter = GuardedMeter([1, 2, 3])
    assert solution["cheat"](meter) == 1
    assert meter.counters.reads == 1 and meter.peek() == [1, 2, 3]


def test_the_state_of_a_collected_object_is_dropped():
    before = len(guarded_module._STATE)
    meter = GuardedMeter([1])
    assert len(guarded_module._STATE) == before + 1
    del meter
    assert len(guarded_module._STATE) == before


# -- an allowlist ----------------------------------------------------------------------


def test_with_an_allowlist_only_the_listed_names_are_readable(solve):
    solution = solve(
        """
        def fine(meter): return meter.read(0) + meter.size
        def other(meter): return meter.last_index
        def klass(meter): return meter.__class__
        """
    )
    meter = ListedMeter([4, 5])
    assert solution["fine"](meter) == 6
    check_guard()
    assert refused(solution["other"], meter) == "`meter.last_index` may not be used in this problem"
    solve("def klass(meter): return meter.__class__")
    assert meter.last_index == 1 and meter.peek() == [4, 5]  # the test still reads everything
    check_guard()


def test_a_subclass_records_its_own_rule(solve):
    class Strict(Guarded, Meter):
        label = "meter"

        def read(self, i):
            if i < 0:
                self.violation("`meter.read` takes no negative index in this problem")
            return Meter.read(self, i)

    solution = solve(
        """
        def cheat(meter):
            try:
                return meter.read(-1)
            except Exception:
                return meter.read(0)
        """
    )
    assert solution["cheat"](Strict([3, 4])) == 3
    with pytest.raises(CheckFailure, match="takes no negative index"):
        check_guard()


def test_a_property_with_a_setter_still_sets():
    class Dial:
        def __init__(self):
            self._level = 0

        @property
        def level(self):
            return self._level

        @level.setter
        def level(self, value):
            self._level = max(0, value)

    class GuardedDial(Guarded, Dial):
        pass

    dial = GuardedDial()
    dial.level = -3
    assert dial.level == 0
    with pytest.raises(CheckFailure, match="may not be copied"):
        copy.copy(dial)
    guarded_module._violations.clear()


# -- special methods --------------------------------------------------------------------


class Shelf:
    """A kit object used through special methods: called, measured, indexed, entered."""

    def __init__(self, values):
        self._values = list(values)
        self.counters = Counters("calls", "reads")
        self.open = False

    def __call__(self, i):
        self.counters.add("calls")
        return self._values[i]

    def __len__(self):
        return len(self._values)

    def __getitem__(self, i):
        self.counters.add("reads")
        return self._values[i]

    def __contains__(self, value):
        return value in self._values

    def __iter__(self):
        return iter(tuple(self._values))

    def __enter__(self):
        self.open = True
        return self

    def __exit__(self, *exc):
        self.open = False

    def __eq__(self, other):  # not in SPECIAL: it stays a borrowed method
        return self._values == other

    __hash__ = None


class GuardedShelf(Guarded, Shelf):
    label = "shelf"


def test_special_methods_are_calls_made_through_the_object(solve):
    solution = solve(
        """
        def use(shelf):
            with shelf as same:
                opened = same.open
            total = shelf(0) + shelf[1] + shelf.__call__(2) + len(shelf) + sum(shelf)
            return total, 3 in shelf, opened, shelf.open, callable(shelf)
        """
    )
    shelf = GuardedShelf([1, 2, 3])
    assert solution["use"](shelf) == (1 + 2 + 3 + 3 + 6, True, True, False, True)
    check_guard()
    assert shelf.counters.snapshot() == {"calls": 2, "reads": 1}
    assert GuardedShelf.__call__.__doc__ == Shelf.__call__.__doc__
    assert shelf(0) == 1 and shelf == [1, 2, 3]  # the test, outside the solution


def test_a_borrowed_special_method_is_refused(solve):
    solution = solve(
        """
        def borrowed(shelf): return type(shelf).__mro__[2].__call__(shelf, 0)
        def compared(shelf): return shelf == [1, 2, 3]
        def through_the_class(shelf): return type(shelf).__call__(shelf, 0)
        """
    )
    shelf = GuardedShelf([1, 2, 3])
    outside = "may not be read outside the shelf's own methods"
    assert refused(solution["borrowed"], shelf) == f"`shelf.counters` {outside}"
    assert refused(solution["compared"], shelf) == f"`shelf._values` {outside}"
    guarded_module._violations.clear()
    assert solution["through_the_class"](shelf) == 1  # the guarded class's own: counted
    check_guard()
    assert shelf.counters.calls == 1  # the borrowed call was refused before it counted


def test_a_special_method_of_the_guarded_class_is_wrapped_once(solve):
    class Logged(Guarded, Shelf):
        label = "shelf"

        def __call__(self, i):
            self.asked = [*getattr(self, "asked", []), i]
            return Shelf.__call__(self, i)

    class Further(Logged):
        pass

    solution = solve("def use(shelf): return shelf(1), len(shelf)")
    for cls in (Logged, Further):
        shelf = cls([4, 5])
        assert solution["use"](shelf) == (5, 2)
        check_guard()
        assert shelf.asked == [1] and shelf.counters.calls == 1
    assert Further.__call__ is Logged.__call__  # inherited, not wrapped again
    assert not callable(GuardedMeter([1]))  # a class without __call__ gets none


def test_of_guards_an_object_that_was_built_elsewhere(solve):
    solution = solve(
        """
        def use(meter): return meter.read(1)
        def cheat(meter): return meter.counters
        """
    )
    plain = Meter([1, 2, 3])
    meter = GuardedMeter.of(plain)
    assert type(meter) is GuardedMeter and solution["use"](meter) == 2
    check_guard()
    assert meter.counters is plain.counters and plain.counters.reads == 1
    assert refused(solution["cheat"], meter) == "`meter.counters` may not be used in this problem"
    again = GuardedMeter.of(meter)  # also from a guarded object
    assert again.counters is plain.counters and again.peek() == [1, 2, 3]


def test_a_guarded_model_and_guarded_experts_are_called(solve, fw):
    class LM(Guarded, TinyLM):
        label = "lm"
        refused = Guarded.refused | {"seed"}

    class Experts(Guarded, CountingExperts):
        label = "experts"
        refused = Guarded.refused | {"seed", "weights", "calls", "reference"}

    solution = solve(
        """
        def honest(lm, experts, tokens, rows):
            logits, cache = lm(tokens)
            logits, cache = lm(tokens[-1:], cache)
            return logits, cache.length, experts(1, rows), lm.vocab_size, experts.n_experts

        def twin(lm): return type(lm)(lm.fw, seed=lm.seed)
        def kit_twin(lm): return type(lm).__mro__[2].with_framework(lm, lm.fw)
        def borrowed(lm, tokens): return type(lm).__mro__[2].__call__(lm, tokens)
        def weights(experts): return experts.weights
        def oracle(experts, rows): return type(experts).__mro__[2].reference(experts, 1, rows)
        def forge(lm): object.__getattribute__(lm, "counters").tokens_processed = 0
        """
    )
    lm, experts = LM(fw, seed=3), Experts(fw, seed=1, n_experts=4, d_model=8, d_hidden=6)
    rows = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    logits, length, out, vocab, n = solution["honest"](lm, experts, [0, 17, 42], fw.tensor(rows))
    check_guard()
    plain = TinyLM(fw, seed=3)
    want, _ = plain([0, 17, 42, 42])
    np.testing.assert_allclose(fw.numpy(logits), fw.numpy(want)[-1:], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fw.numpy(out), experts.reference(1, rows), rtol=1e-6)
    assert (length, vocab, n) == (4, 64, 4)
    assert lm.counters.snapshot() == {"forward_calls": 2, "tokens_processed": 4}
    assert experts.counters.snapshot() == {"expert_calls": 1, "expert_rows": 5}
    assert isinstance(lm.with_framework(fw), TinyLM)  # the test may make a twin

    assert refused(solution["twin"], lm) == "`lm.seed` may not be used in this problem"
    assert "`lm.__dict__` may not be read outside" in refused(solution["kit_twin"], lm)
    assert "`lm._token_array` may not be read outside" in refused(solution["borrowed"], lm, [0, 1])
    assert refused(solution["weights"], experts) == (
        "`experts.weights` may not be used in this problem"
    )
    assert refused(solution["oracle"], experts, rows) == (
        "`experts._expert` may not be read outside the experts' own methods"
    )
    with pytest.raises(AttributeError):
        solution["forge"](lm)
    assert lm.counters.tokens_processed == 4


# -- the kit's own objects behind the guard -------------------------------------------


@pytest.fixture(params=["torch", "jax", "mlx"])
def fw(request):
    try:
        return frameworks.get(request.param)
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))


def test_a_guarded_store_counts_and_stays_guarded_when_appended(solve, fw):
    class Store(Guarded, CountingKV):
        label = "store"

    solution = solve(
        """
        def honest(store, rows, k, v):
            longer = store.append(k, v)
            return longer.gather(rows), longer.n, type(longer).__name__

        def twin(store): return store.with_framework(store.fw)
        def arrays(store): return type(store).__mro__[2].reference(store)
        def forge(store): object.__getattribute__(store, "counters").rows_read = 0
        """
    )
    rng = np.random.default_rng(0)
    k, v = rng.normal(size=(4, 3)).astype(np.float32), rng.normal(size=(4, 2)).astype(np.float32)
    store = Store(fw, k, v)
    (k_rows, _), n, name = solution["honest"](store, [0, 5], k[:2], v[:2])
    check_guard()
    assert (n, name) == (6, "Store")
    np.testing.assert_array_equal(fw.numpy(k_rows), np.stack((k[0], k[1])))
    assert store.counters.rows_read == 2 and store.log == [((2,), 2)]
    assert isinstance(store.with_framework(fw), CountingKV)  # the test may make a twin

    assert (
        refused(solution["twin"], store) == "`store.with_framework` may not be used in this problem"
    )
    solve("def arrays(store): return type(store).__mro__[2].reference(store)")
    with pytest.raises(CheckFailure, match="`store._k` may not be read outside the store's own"):
        solution["arrays"](store)
    with pytest.raises(AttributeError):
        solution["forge"](store)
    assert store.counters.rows_read == 2


def test_guarded_models_run_their_oracles_and_refuse_their_internals(solve, fw):
    class Layers(Guarded, TinyDSA):
        label = "dsa"

    class Model(Guarded, BlockDenoiser):
        label = "model"
        allowed = frozenset({"encode", "denoise", "vocab_size", "eos_id", "pad_id"})

    solution = solve(
        """
        def keys(dsa, x): return dsa.index_key(0, x)
        def free_keys(dsa, x): return type(dsa).__mro__[2]._index_key(dsa, 0, x)
        def logits(model, canvas): return model.denoise(canvas, model.encode([3, 4]))
        def weights(model): return type(model).__mro__[2]._denoise(model, [1], None, None)
        def seed(model): return model.seed
        """
    )
    dsa = Layers(fw, seed=1)
    x = np.random.default_rng(1).normal(size=(5, dsa.d_model)).astype(np.float32)
    plain = TinyDSA(fw, seed=1)
    np.testing.assert_allclose(
        fw.numpy(solution["keys"](dsa, fw.tensor(x))), fw.numpy(plain.index_key(0, fw.tensor(x)))
    )
    check_guard()
    assert dsa.counters.index_key_rows == 5
    np.testing.assert_allclose(dsa.layer_ref(0, x), plain.layer_ref(0, x))  # the test's oracle
    assert refused(solution["free_keys"], dsa, x).startswith("`dsa._")
    assert dsa.counters.index_key_rows == 5

    guarded_module._violations.clear()
    model = Model(fw, seed=2)
    canvas = fw.tensor(np.array([1, 2, 3], dtype=np.int64))
    got = fw.numpy(solution["logits"](model, canvas))
    assert got.shape == (3, model.vocab_size) and model.counters.snapshot()["denoise_calls"] == 1
    check_guard()
    assert refused(solution["seed"], model) == "`model.seed` may not be used in this problem"
    assert "outside the model's own methods" in refused(solution["weights"], model)


# -- DeltaMemory -----------------------------------------------------------------------


def test_delta_memory_reads_and_updates_in_counted_calls(fw):
    rng = np.random.default_rng(3)
    state = rng.normal(size=(2, 3, 4)).astype(np.float32)  # H, Dv, Dk
    memory = DeltaMemory(fw, fw.tensor(state), max_rows=5)
    assert memory.shape == (2, 3, 4) and memory.max_rows == 5

    rows = rng.normal(size=(2, 5, 4)).astype(np.float32)
    out = memory.read(fw.tensor(rows))
    assert fw.is_tensor(out)
    np.testing.assert_allclose(fw.numpy(out), rows @ state.transpose(0, 2, 1), rtol=1e-6)

    decay = np.array([0.5, 1.0], np.float32)
    values, keys = rng.normal(size=(2, 2, 3)), rng.normal(size=(2, 2, 4))
    values, keys = values.astype(np.float32), keys.astype(np.float32)
    newer = memory.update(fw.tensor(decay), fw.tensor(values), fw.tensor(keys))
    want = state * decay[:, None, None] + values.transpose(0, 2, 1) @ keys
    np.testing.assert_allclose(newer.peek(), want, rtol=1e-6)
    np.testing.assert_array_equal(memory.peek(), state)  # the old memory is unchanged
    assert newer.counters is memory.counters
    assert memory.counters.snapshot() == {"reads": 1, "writes": 1}  # one per call, any R
    newer.peek()[...] = 0
    np.testing.assert_allclose(newer.peek(), want, rtol=1e-6)  # peek hands out a copy

    twin = newer.with_framework(fw)
    np.testing.assert_array_equal(twin.peek(), newer.peek())
    assert twin.counters.snapshot() == {"reads": 0, "writes": 0}
    assert "reads=1" in repr(memory)


def test_delta_memory_refuses_wrong_shapes_and_too_many_rows(fw):
    memory = DeltaMemory(fw, np.zeros((2, 3, 4), np.float32), max_rows=2)
    ones = np.ones
    with pytest.raises(ValueError, match=r"rows must have shape \(2, R, 4\)"):
        memory.read(ones((2, 1, 3), np.float32))
    with pytest.raises(ValueError, match="1 to max_rows = 2 rows, got 3"):
        memory.read(ones((2, 3, 4), np.float32))
    with pytest.raises(ValueError, match="1 to max_rows = 2 rows, got 0"):
        memory.read(ones((2, 0, 4), np.float32))
    with pytest.raises(ValueError, match="decay must have shape"):
        memory.update(ones(3, np.float32), ones((2, 1, 3), np.float32), ones((2, 1, 4), np.float32))
    with pytest.raises(ValueError, match="the same number of rows"):
        memory.update(ones(2, np.float32), ones((2, 1, 3), np.float32), ones((2, 2, 4), np.float32))
    with pytest.raises(ValueError, match="state must have shape"):
        DeltaMemory(fw, np.zeros((3, 4), np.float32), max_rows=2)
    assert memory.counters.snapshot() == {"reads": 0, "writes": 0}  # a refused call is free


def test_a_guarded_delta_memory_keeps_its_array_and_counters_to_itself(solve, fw):
    class Memory(Guarded, DeltaMemory):
        label = "memory"
        refused = Guarded.refused | {"peek"}

    solution = solve(
        """
        def honest(memory, rows, decay, values, keys):
            return memory.read(rows), memory.update(decay, values, keys), memory.max_rows

        def peek(memory): return memory.peek()
        def borrowed(memory): return type(memory).__mro__[2].peek(memory)
        def state(memory): return object.__getattribute__(memory, "_state")
        """
    )
    memory = Memory(fw, np.ones((1, 2, 3), np.float32), max_rows=4)
    rows = np.ones((1, 2, 3), np.float32)
    _, newer, max_rows = solution["honest"](
        memory, rows, np.ones(1, np.float32), np.ones((1, 2, 2), np.float32), rows
    )
    check_guard()
    assert isinstance(newer, Memory) and newer.counters is memory.counters and max_rows == 4
    np.testing.assert_array_equal(newer.peek(), np.full((1, 2, 3), 3.0, np.float32))
    assert refused(solution["peek"], memory) == "`memory.peek` may not be used in this problem"
    assert "`memory._state` may not be read outside" in refused(solution["borrowed"], newer)
    with pytest.raises(AttributeError):
        solution["state"](memory)
