"""Tests for device-map-solver.

The rule set of the statement determines the map, so every case compares the
solution's dict with a literal or with the oracle's. The property cases also check,
independently of the oracle, what a device map must satisfy whatever rule produced
it: every weight placed exactly once, no device over its budget, no-split modules
whole, devices in forward order, tied modules next to their owners.
"""

from learnai.judge import case, check

GB = 1 << 30


def block(name, attn, mlp, cls="Block"):
    return (name, cls, [(f"{name}.attn", "Attention", attn), (f"{name}.mlp", "MLP", mlp)])


def show(modules, indent=2):
    lines = []
    for name, cls, body in modules:
        if isinstance(body, int):
            lines.append(f"{' ' * indent}({name!r}, {cls!r}, {body})")
        else:
            lines.append(f"{' ' * indent}({name!r}, {cls!r}, [")
            lines.extend(show(body, indent + 4))
            lines.append(f"{' ' * indent}])")
    return lines


def describe(modules, max_memory, no_split, tied):
    return (
        "modules = [\n" + "\n".join(show(modules)) + "\n]\n"
        f"max_memory = {max_memory!r}\nno_split = {sorted(no_split)!r}\ntied = {tied!r}\n"
    )


def solve(ctx, modules, max_memory, no_split, tied):
    got = ctx.solution.solve_device_map(modules, dict(max_memory), set(no_split), dict(tied))
    check.that(
        isinstance(got, dict),
        f"solve_device_map returned a {type(got).__name__}, which is not a dict",
    )
    return got


def check_map(ctx, modules, max_memory, no_split, tied, expected=None, label="device map"):
    got = solve(ctx, modules, max_memory, no_split, tied)
    if expected is None:
        expected = ctx.oracle.solve_device_map(modules, dict(max_memory), set(no_split), dict(tied))
    detail = describe(modules, max_memory, no_split, tied)
    detail += f"expected: {expected!r}\n     got: {got!r}\n"
    check.that(
        set(got) == set(expected),
        f"{label}: the modules named in the map are not the ones the rules place",
        detail,
    )
    wrong = sorted(name for name in got if got[name] != expected[name])
    check.that(
        not wrong,
        f"{label}: {len(wrong)} of {len(got)} modules are on the wrong device "
        f"(for example {wrong[0]!r})" if wrong else "",
        detail,
    )  # fmt: skip
    return got


# -- invariants, computed without the oracle ------------------------------------------------


def size(module):
    body = module[2]
    return body if isinstance(body, int) else sum(size(child) for child in body)


def walk(modules, ancestors=()):
    """Every module in forward (pre-)order, with the names of its ancestors."""
    for module in modules:
        yield module, ancestors
        if not isinstance(module[2], int):
            yield from walk(module[2], (*ancestors, module[0]))


def check_invariants(ctx, modules, max_memory, no_split, tied, got):
    detail = describe(modules, max_memory, no_split, tied) + f"got: {got!r}\n"
    nodes = list(walk(modules))
    known = {module[0]: module for module, _ in nodes}
    devices = [*max_memory, "disk"]
    rank = {device: i for i, device in enumerate(devices)}

    unknown = [name for name in got if name not in known]
    check.that(not unknown, "the map names something that is not a module of the model", detail)
    bad = [name for name, device in got.items() if device not in rank]
    check.that(not bad, "the map uses a device that is neither in max_memory nor 'disk'", detail)

    # Every leaf is covered by exactly one entry: itself or one ancestor.
    placed_order = []
    for module, ancestors in nodes:
        if module[0] in got:
            placed_order.append(module[0])
        if isinstance(module[2], int):
            covering = [name for name in (*ancestors, module[0]) if name in got]
            check.that(
                len(covering) == 1,
                "a weight is placed "
                + ("nowhere" if not covering else "twice (a module and one of its ancestors)"),
                detail + f"leaf {module[0]!r} is covered by {covering!r}\n",
            )
    for module, ancestors in nodes:
        if module[1] in no_split and not isinstance(module[2], int):
            whole = any(name in got for name in (*ancestors, module[0]))
            check.that(
                whole,
                "a module of a no-split class was divided between entries of the map",
                detail + f"{module[0]!r} has class {module[1]!r}\n",
            )

    reserve = max(
        (size(m) for m, _ in nodes if isinstance(m[2], int) or m[1] in no_split), default=0
    )
    used = dict.fromkeys(devices, 0)
    for name, device in got.items():
        used[device] += size(known[name])
    for i, device in enumerate(max_memory):
        limit = max_memory[device] - (reserve if i == 0 else 0)
        check.that(
            used[device] <= limit or used[device] == 0,
            f"device {device!r} holds more bytes than its budget allows",
            detail + f"device {device!r}: {used[device]} bytes placed, budget {limit}"
            + (f" (max_memory minus the reserve of {reserve})" if i == 0 else "") + "\n",
        )  # fmt: skip

    forward = [name for name in placed_order if name not in tied]
    ranks = [rank[got[name]] for name in forward]
    back = next((i for i in range(1, len(ranks)) if ranks[i] < ranks[i - 1]), None)
    check.that(
        back is None,
        "the devices are not in forward order: a later module sits on an earlier device",
        "" if back is None else detail + f"{forward[back - 1]!r} is on {got[forward[back - 1]]!r} "
        f"but the later {forward[back]!r} is on {got[forward[back]]!r}\n",
    )  # fmt: skip

    for name, owner in tied.items():
        holder = next(
            n for m, anc in nodes if m[0] == owner for n in (*anc, owner) if n in got
        )  # fmt: skip
        check.that(
            got.get(name) == got[holder],
            "a tied module is not on the device of the weights it shares",
            detail + f"{name!r} is tied to {owner!r}, which is on {got[holder]!r}\n",
        )


# -- model builders ----------------------------------------------------------------------------


def decoder(n_layers, *, embed, attn, mlp, norm=1, head=0, block_cls="Block"):
    """A causal LM as transformers lays it out: model.{embed_tokens, layers, norm}, lm_head."""
    layers = [block(f"model.layers.{i}", attn, mlp, block_cls) for i in range(n_layers)]
    return [
        ("model", "Decoder", [
            ("model.embed_tokens", "Embedding", embed),
            ("model.layers", "ModuleList", layers),
            ("model.norm", "Norm", norm),
        ]),
        ("lm_head", "Linear", head),
    ]  # fmt: skip


def random_model(rng):
    """A random module tree with no-split classes and tied top-level leaves."""
    classes = ["Block", "Stage", "Linear", "Norm", "Conv", "Adapter"]
    counter = [0]

    def make(prefix, depth):
        counter[0] += 1
        name = f"{prefix}m{counter[0]}"
        cls = classes[int(rng.integers(len(classes)))]
        if depth >= 3 or rng.random() < 0.45:
            return (name, cls, int(rng.choice([0, 1, 2, 3, 5, 8, 13, 21, 40])))
        children = [make(name + ".", depth + 1) for _ in range(int(rng.integers(1, 5)))]
        return (name, cls, children)

    modules = [make("", 0) for _ in range(int(rng.integers(2, 7)))]
    no_split = {cls for cls in classes if rng.random() < 0.3}
    tied = {}
    leaves_before = []
    for module in modules:
        is_leaf = isinstance(module[2], int)
        if is_leaf and leaves_before and rng.random() < 0.3:
            tied[module[0]] = leaves_before[int(rng.integers(len(leaves_before)))]
        else:
            leaves_before.extend(m[0] for m, _ in walk([module]) if isinstance(m[2], int))
    total = sum(size(m) for m in modules)
    names = [0, 1, 2, "cpu"][: int(rng.integers(0, 5))]
    max_memory = {d: int(rng.integers(0, max(2, total))) for d in names}
    if max_memory and rng.random() < 0.5:
        first = next(iter(max_memory))
        max_memory[first] += int(rng.integers(0, 60))
    return modules, max_memory, no_split, tied


# -- cases -------------------------------------------------------------------------------------

EXAMPLE = [
    ("embed", "Embedding", 400),
    ("layers", "ModuleList", [block(f"layers.{i}", 300, 500) for i in range(3)]),
    ("norm", "Norm", 10),
    ("lm_head", "Linear", 0),
]


@case("the worked example: two GPUs and the CPU, a reserve, a tied head", public=True)
def worked_example(ctx):
    check_map(
        ctx, EXAMPLE, {0: 2000, 1: 1000, "cpu": 900}, {"Block"}, {"lm_head": "embed"},
        {"embed": 0, "lm_head": 0, "layers.0": 0, "layers.1": 1, "layers.2": "cpu", "norm": "cpu"},
    )  # fmt: skip


@case("a module that fits is placed whole; a container is opened only when it does not",
      public=True)  # fmt: skip
def whole_when_it_fits(ctx):
    # Rejects always_splits. The reserve is 800 (one Block).
    check_map(
        ctx, EXAMPLE, {0: 4000}, {"Block"}, {},
        {"embed": 0, "layers": 0, "norm": 0, "lm_head": 0},
        label="everything fits on device 0",
    )  # fmt: skip
    check_map(
        ctx, EXAMPLE, {0: 3610, "cpu": 100}, {"Block"}, {},
        {"embed": 0, "layers": 0, "norm": 0, "lm_head": 0},
        label="an exact fit: 2810 bytes in a budget of 3610 - 800",
    )  # fmt: skip
    check_map(
        ctx, EXAMPLE, {0: 3609, "cpu": 100}, {"Block"}, {},
        {"embed": 0, "layers": 0, "norm": "cpu", "lm_head": "cpu"},
        label="one byte short: `norm` moves on, and so does everything after it",
    )  # fmt: skip
    check_map(
        ctx, EXAMPLE, {0: 3000, "cpu": 900}, {"Block"}, {},
        {"embed": 0, "layers.0": 0, "layers.1": 0, "layers.2": "cpu", "norm": "cpu",
         "lm_head": "cpu"},
        label="`layers` does not fit: it is opened and its last block moves on",
    )  # fmt: skip


@case("no-split classes stay whole; other modules are divided down to their leaves",
      public=True)  # fmt: skip
def no_split_and_split(ctx):
    # Rejects splits_no_split and never_splits.
    limits = {0: 2500, 1: 1000}
    check_map(
        ctx, EXAMPLE, limits, {"Block"}, {},
        {"embed": 0, "layers.0": 0, "layers.1": 1, "layers.2": "disk", "norm": "disk",
         "lm_head": "disk"},
        label="Block is no-split",
    )  # fmt: skip
    # Nothing is no-split: the reserve is the largest leaf (500), and blocks are divided.
    check_map(
        ctx, EXAMPLE, limits, set(), {},
        {"embed": 0, "layers.0": 0, "layers.1": 0, "layers.2.attn": 1, "layers.2.mlp": 1,
         "norm": 1, "lm_head": 1},
        label="nothing is no-split",
    )  # fmt: skip
    check_map(
        ctx, EXAMPLE, {0: 1700, 1: 1150}, set(), {},
        {"embed": 0, "layers.0": 0, "layers.1.attn": 1, "layers.1.mlp": 1, "layers.2.attn": 1,
         "layers.2.mlp": "disk", "norm": "disk", "lm_head": "disk"},
        label="a block divided between device 1 and disk",
    )  # fmt: skip


@case("tied weights: the head goes where the embedding went, and its bytes count there",
      public=True)  # fmt: skip
def tied_weights(ctx):
    # Rejects tied_ignored and tied_size_ignored.
    model = decoder(4, embed=600, attn=200, mlp=300, norm=1, head=40)
    tied = {"lm_head": "model.embed_tokens"}
    # Reserve 600 (the embedding is the largest unit). Budget on 0: 2000 - 600 = 1400.
    check_map(
        ctx, model, {0: 2000, 1: 5000}, {"Block"}, tied,
        {"model.embed_tokens": 0, "lm_head": 0, "model.layers.0": 0, "model.layers.1": 1,
         "model.layers.2": 1, "model.layers.3": 1, "model.norm": 1},
        label="the owner is inside `model`, which is opened",
    )  # fmt: skip
    check_map(
        ctx,
        model,
        {0: 4000, 1: 5000},
        {"Block"},
        tied,
        {"model": 0, "lm_head": 0},
        label="the owner is placed inside `model`, whole",
    )
    # 600 + 40 does not fit in 1239 - 600; the embedding alone would have.
    check_map(
        ctx, model, {0: 1239, 1: 1700}, {"Block"}, tied,
        {"model.embed_tokens": 1, "lm_head": 1, "model.layers.0": 1, "model.layers.1": 1,
         "model.layers.2": "disk", "model.layers.3": "disk", "model.norm": "disk"},
        label="the owner and the tied module must fit together",
    )  # fmt: skip


@case("overflow to the CPU and to disk; a device that was left is never used again")
def overflow_in_order(ctx):
    # Rejects first_fit (a small later module on an earlier device) and no_disk.
    model = [
        ("stem", "Conv", 100),
        ("big", "Stage", 900),
        ("small", "Linear", 50),
        ("mid", "Stage", 400),
        ("tail", "Norm", 5),
        ("huge", "Stage", 5000),
        ("after", "Norm", 1),
    ]
    check_map(
        ctx, model, {0: 5500, 1: 950, "cpu": 420}, set(), {},
        {"stem": 0, "big": 1, "small": 1, "mid": "cpu", "tail": "cpu", "huge": "disk",
         "after": "disk"},
    )  # fmt: skip
    check_map(ctx, model, {}, set(), {}, dict.fromkeys((m[0] for m in model), "disk"),
              label="device map")  # fmt: skip
    check_map(ctx, model, {0: 4999}, set(), {}, dict.fromkeys((m[0] for m in model), "disk"),
              label="device map")  # fmt: skip


@case("the reserve: the largest indivisible unit, on the first device only")
def reserve_rules(ctx):
    # Rejects no_reserve, reserve_everywhere, reserve_top_level, and strict_fit.
    model = decoder(6, embed=300, attn=100, mlp=250, norm=2, head=300)
    for no_split in ({"Block"}, set(), {"Decoder"}, {"ModuleList"}):
        for limits in ({0: 1200, 1: 1200, "cpu": 700}, {0: 950, 1: 351, "cpu": 349},
                       {"cpu": 2000}, {0: 650, 1: 650, 2: 650, 3: 650}):  # fmt: skip
            check_map(ctx, model, limits, no_split, {},
                      label="device map")  # fmt: skip


@case("a 32-layer decoder in bytes on two 24 GB GPUs with CPU offload")
def realistic(ctx):
    # A 13 B model in fp16: hidden 5120, vocabulary 32000, about 0.8 GB per layer.
    model = decoder(
        32, embed=32000 * 5120 * 2, attn=4 * 5120 * 5120 * 2, mlp=3 * 5120 * 13824 * 2,
        norm=5120 * 2, head=0, block_cls="LlamaDecoderLayer",
    )  # fmt: skip
    tied = {"lm_head": "model.embed_tokens"}
    for limits in ({0: 24 * GB, 1: 24 * GB, "cpu": 64 * GB}, {0: 22 * GB, "cpu": 8 * GB},
                   {0: 10 * GB, 1: 10 * GB}, {0: 80 * GB}):  # fmt: skip
        got = check_map(ctx, model, limits, {"LlamaDecoderLayer"}, tied, label="device map")
        check_invariants(ctx, model, limits, {"LlamaDecoderLayer"}, tied, got)


@case("random module trees: budgets respected, forward order, no-split whole, ties kept")
def invariants_hold(ctx):
    def prop(rng):
        modules, max_memory, no_split, tied = random_model(rng)
        got = solve(ctx, modules, max_memory, no_split, tied)
        check_invariants(ctx, modules, max_memory, no_split, tied, got)

    check.forall(prop, trials=300)


@case("random module trees: the map is exactly the one the rules give")
def random_exact(ctx):
    def prop(rng):
        modules, max_memory, no_split, tied = random_model(rng)
        check_map(ctx, modules, max_memory, no_split, tied)

    check.forall(prop, trials=300)
