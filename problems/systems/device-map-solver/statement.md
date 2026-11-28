Implement `solve_device_map`, which assigns each part of a model to a device when the model does not fit on a single GPU. It walks the model's modules in forward order and fills devices sequentially (typically GPUs first, then CPU memory, and finally disk) while respecting indivisible blocks, a memory reserve on the first device, and tied weights.

## Specification

```python
def solve_device_map(
    modules: list[tuple],
    max_memory: dict[int | str, int],
    no_split: set[str],
    tied: dict[str, str],
) -> dict[str, int | str]: ...
```

**The model.** `modules` lists the top-level modules in forward-pass order. Each module is a tuple `(name, cls, body)`:

- `name`: a unique string, such as `"model.layers.3.mlp"`.
- `cls`: the module's class name, such as `"LlamaDecoderLayer"`.
- `body`: either an `int` for a **leaf** (the size of its weights in bytes, possibly 0) or a non-empty list of child module tuples in forward order for a **container**. A container has no weights of its own, so its **size** is the sum of its children's sizes.

A module is **indivisible** if it is a leaf or if its `cls` is in `no_split`. Blocks with residual connections belong in `no_split` because splitting them across devices would require the skip tensor to live on both devices.

**The devices.** `max_memory` maps each device to the number of bytes it can hold. Its key order determines the order in which devices are filled (for example, GPUs `0, 1, ...`, then `"cpu"`). `"disk"` is never a key in `max_memory`. It always comes last and has unlimited capacity.

**The reserve.** The first device in `max_memory` also holds model inputs and temporarily stages weights brought in from slower devices for execution. The **reserve** is the size of the largest indivisible module anywhere in the model (across all leaves and all modules whose `cls` is in `no_split`, at any depth). To keep space free, the first device's **budget** is `max_memory[first]` minus this reserve. Every other device has its full `max_memory` value as its budget. Compute the reserve once before placing any module.

**Tied weights.** An entry `tied[t] = o` means that the top-level leaf `t` shares its weights with its **owner** leaf `o` (for example, an output head tied to an input embedding). The size of `t` is the extra bytes it needs beyond `o` (often 0). Never place `t` on its own. Instead, place `t` **together with the module that contains `o`**, on the same device and at the same moment. The **followers** of a module `m` are all tied leaves whose owner is `m` or lies inside `m`.

**The algorithm.** Maintain a queue of modules (initially the top-level modules that are not keys of `tied`, in order), a current device (initially the first device, or `"disk"` if `max_memory` is empty), and the bytes `used` on the current device (initially 0). While the queue is not empty, remove the module `m` at its front and let `need` be the size of `m` plus the sizes of its followers:

1. **Fit.** If the current device is `"disk"`, or if `used + need <= budget`, `m` fits. Map `m`'s name and the name of each follower to the current device, and increase `used` by `need`.
2. **Open.** Otherwise, if `m` is not indivisible, open it. Insert its children at the front of the queue in forward order and stay on the current device.
3. **Advance.** Otherwise, close the current device. Move to the next device with `used = 0` and try `m` again there. A closed device is never used again, even if a later module would fit in its remaining space.

Return the resulting map. When a container is placed whole, include only its own name (not its children), so every leaf is covered by exactly one entry. The order of the dict's keys does not matter.

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

![The example's devices, each bar filled in module order. Device 0 keeps 800 bytes in reserve, so its budget is 1200.](assets/device-map.svg)

The largest indivisible module is a `Block` of 800 bytes, so device 0 has a budget of $2000 - 800 = 1200$.

| front of the queue | need | device, used before | outcome |
|---|---|---|---|
| `embed` (+ `lm_head`) | 400 + 0 | 0, 0 | fits (`lm_head` placed with it) |
| `layers` | 2400 | 0, 400 | does not fit (`ModuleList`), opened |
| `layers.0` | 800 | 0, 400 | fits exactly (1200) |
| `layers.1` | 800 | 0, 1200 | does not fit (`Block`), close device 0 |
| `layers.1` | 800 | 1, 0 | fits |
| `layers.2` | 800 | 1, 800 | does not fit, close device 1 |
| `layers.2` | 800 | cpu, 0 | fits |
| `norm` | 10 | cpu, 800 | fits |

With `no_split = set()`, the reserve would be 500 (the largest leaf), and a block that does not fit would be opened. Its attention and MLP submodules could then land on different devices.

## Constraints

- Up to 2000 modules in total, nested at most 6 levels deep. Sizes and limits are ints in $[0, 2^{50}]$. `max_memory` has 0 to 8 devices (with 0 devices, everything goes to `"disk"`).
- Every key of `tied` is a top-level leaf. Its owner is a leaf (at any depth) that is not itself tied and belongs to an earlier top-level module.
- A budget can be zero or negative when the reserve exceeds the first device's capacity. Nothing fits on that device, not even a 0-byte module, and the first indivisible module tried there closes it.
- The rules determine the map uniquely, and tests compare it exactly. Independently, property tests on random models verify that every leaf is covered once, no device exceeds its budget, no-split modules remain whole, devices never go backwards along forward order (excluding tied modules), and each tied module shares its owner's device.
- Do not modify the input arguments.

## Why this matters

A model split across devices runs one piece at a time, moving activations from device to device in layer order. A map that fits within memory limits but places layer 20 back on GPU 0 after layer 19 on GPU 1 pays an extra cross-device transfer in both directions on every forward pass.

Other placement bugs fail even harder. Filling GPU 0 to the last byte leaves no room for inputs or offloaded weights, causing an out-of-memory crash on the first real batch. Splitting a block across its residual connection fails outright because tensors end up on different devices. Layers that spill onto CPU memory or disk still execute, but at a tiny fraction of GPU speed, so checking where each layer landed is essential before serving a model.
