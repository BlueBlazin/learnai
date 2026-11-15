Decide where each part of a model lives when it does not fit on one GPU: walk the
modules in forward order and fill the devices one after another, GPUs first, then
CPU memory, then disk. This is what `device_map="auto"` computes before any weight
is loaded. The loop is short; the rules that make the answer usable are when a
module may be divided, how much of the first device to leave free, and what happens
to weights that two modules share.

## Specification

```python
def solve_device_map(
    modules: list[tuple],
    max_memory: dict[int | str, int],
    no_split: set[str],
    tied: dict[str, str],
) -> dict[str, int | str]: ...
```

**The model.** `modules` lists the top-level modules in the order the forward pass
uses them. A module is a tuple `(name, cls, body)`:

- `name`: a unique string, for example `"model.layers.3.mlp"`.
- `cls`: the name of its class, for example `"LlamaDecoderLayer"`.
- `body`: an `int` for a **leaf** (the bytes of its weights, possibly 0), or a
  non-empty list of child modules, in forward order, for a **container**. A
  container has no weights of its own: its **size** is the sum of its children's
  sizes.

A module is **indivisible** if it is a leaf or if its `cls` is in `no_split`
(a block with a residual connection must not be cut in two: the tensor that skips
around it would have to be on both devices).

**The devices.** `max_memory` maps each device to the bytes it may hold, and its key
order is the order in which the devices are filled (GPUs `0, 1, ...`, then `"cpu"`,
by the caller's choice). The device `"disk"` is never a key: it always comes last
and has no limit.

**The reserve.** The first device of `max_memory` also holds the inputs and whatever
is brought in from slower devices to be executed, so part of it stays free: its
budget is `max_memory[first]` minus the **reserve**, the size of the largest
indivisible module anywhere in the model (among all leaves and all modules of a
no-split class, at any depth). Every other device has its full `max_memory` as
budget. The reserve is computed once, before anything is placed.

**Tied weights.** `tied[t] = o` says that the top-level leaf `t` shares its weights
with the leaf `o` (an output head tied to the embedding). The size of `t` is what it
needs in addition (often 0). `t` is never placed on its own: it is placed
**together with the module that contains `o`**, on the same device and at the same
moment. The *followers* of a module `m` are the tied leaves whose owner is `m` or
lies inside `m`.

**The algorithm.** Keep a queue of modules, initially the top-level modules that are
not keys of `tied`, in order; a current device, initially the first; and the bytes
used on it, initially 0. Until the queue is empty, take the module `m` at its front
and let `need` be the size of `m` plus the sizes of its followers.

1. If the current device is `"disk"`, or `used + need <= budget`: `m` **fits**. The
   map gets `m`'s name and the name of each follower, all with the current device,
   and `used` grows by `need`.
2. Otherwise, if `m` is not indivisible: **open** it. Its children take its place at
   the front of the queue, in order, and the current device stays.
3. Otherwise: the current device is **closed**. The next device becomes current with
   `used = 0`, and `m` is tried again there. A closed device is never used again,
   even if a later module would fit in what is left of it.

Return the map. It contains a module that was placed whole under its own name (not
its children), so every leaf is covered by exactly one entry. The order of the
dict's keys does not matter.

## Example

```python
def block(i):
    return (
        f"layers.{i}",
        "Block",
        [(f"layers.{i}.attn", "Attention", 300), (f"layers.{i}.mlp", "MLP", 500)],
    )


modules = [
    ("embed", "Embedding", 400),
    ("layers", "ModuleList", [block(0), block(1), block(2)]),
    ("norm", "Norm", 10),
    ("lm_head", "Linear", 0),
]
solve_device_map(modules, {0: 2000, 1: 1000, "cpu": 900}, {"Block"}, {"lm_head": "embed"})
# {"embed": 0, "lm_head": 0, "layers.0": 0, "layers.1": 1, "layers.2": "cpu", "norm": "cpu"}
```

The largest indivisible module is a `Block` of 800 bytes, so device 0 has a budget
of $2000 - 800 = 1200$.

| front of the queue | need | device, used before | outcome |
|---|---|---|---|
| `embed` (+ `lm_head`) | 400 + 0 | 0, 0 | fits; `lm_head` goes with it |
| `layers` | 2400 | 0, 400 | does not fit, a `ModuleList`: opened |
| `layers.0` | 800 | 0, 400 | fits exactly (1200) |
| `layers.1` | 800 | 0, 1200 | does not fit, a `Block`: device 0 is closed |
| `layers.1` | 800 | 1, 0 | fits |
| `layers.2` | 800 | 1, 800 | does not fit: device 1 is closed |
| `layers.2` | 800 | cpu, 0 | fits |
| `norm` | 10 | cpu, 800 | fits |

With `no_split = set()` the reserve would be 500 (the largest leaf), and a block
that does not fit would be opened: its attention and its MLP could land on
different devices.

## Constraints

- Up to 2000 modules in total, nested at most 6 deep. Sizes and limits are ints in
  $[0, 2^{50}]$. `max_memory` has 0 to 8 devices (with none, everything is on
  `"disk"`).
- Every key of `tied` is a top-level leaf; its owner is a leaf (at any depth) that
  is not itself tied and belongs to an earlier top-level module.
- A budget can be zero or negative (a reserve larger than the first device). Then
  nothing fits there, not even a module of size 0, and the device is closed by the
  first indivisible module that is tried on it.
- The rules determine the map; it is compared exactly. Independently of that, the
  tests check on random models that every leaf is covered once, that no device
  holds more than its budget, that no-split modules are whole, that devices never
  go backwards along the forward order (tied modules aside), and that each tied
  module is on the device of its owner.
- Do not modify the arguments.

## Why this matters

A model split over devices runs one piece at a time, and the activations travel
from device to device in the order of the layers. A map that respects the memory
limits but puts layer 20 back on GPU 0 after layer 19 on GPU 1 pays a transfer in
each direction on every forward pass; a map that fills GPU 0 to the last byte runs
out of memory at the first real batch, because the weights were not the only thing
that had to fit there; a block cut through its residual connection fails outright
with tensors on two devices. And layers that fall off the last GPU still run, from
CPU memory or from disk, at a small fraction of the speed, which is why "it loaded"
and "it is usable" are different statements about a device map.
