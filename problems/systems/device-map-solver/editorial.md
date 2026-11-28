## The idea

A device map partitions a model's weights into consecutive runs, one per device. Keeping runs consecutive is critical. During a forward pass, activations move to wherever the next module lives, so the number of device-to-device transfers equals the number of device boundaries in the map. Filling devices strictly in order minimizes transfers to one boundary per device and reduces placement to a single forward scan. At each step, there is one current device, and the only question is whether the next module fits.

This greedy scan does not maximize GPU utilization. A bin-packing solver that tucks a small late layer into leftover space on GPU 0 holds more weights on GPUs, but runs slower because activations must visit GPU 0 twice. Leaving a few hundred megabytes unused at the end of a device is the cost of preserving forward order.

**What may be divided.** The scan operates on a module tree. A module that fits is placed whole, which keeps the map compact. A container that does not fit is opened so its children can be tried on the same device, splitting a stack of layers cleanly between two layers. Recursion stops at leaves and at modules whose class is in `no_split`. A transformer block computes `x + f(x)`. If the block were split, `x` would live on the starting device while `f(x)` lived on the next, causing the residual addition to fail.

**The reserve.** Weights are not the only data a device holds. The first device also receives inputs, holds outputs during post-processing, and executes modules offloaded to CPU or disk by copying their weights in just before each forward call and dropping them afterward. The largest module that might be staged in must fit alongside the weights permanently stored on the first device. Reserving the size of the largest indivisible module guarantees that room.

**Tied weights.** An output head tied to an input embedding is a single tensor under two names. Placed independently, the embedding would land on the first device and the head on the last, forcing the loader to duplicate the tensor or break the tie. Instead, each tied module travels with its owner. We exclude it from the initial queue, add its extra bytes to its owner's size check, and assign both to the same device. This is the one place where forward execution moves backwards across devices, from the last layer on a later device back to the head on GPU 0.

## Reference solution (Python)

```python
def solve_device_map(modules, max_memory, no_split, tied):
    def size(module):
        body = module[2]
        return body if isinstance(body, int) else sum(size(child) for child in body)

    def atomic(module):
        return isinstance(module[2], int) or module[1] in no_split

    def leaf_names(module):
        if isinstance(module[2], int):
            return [module[0]]
        return [leaf for child in module[2] for leaf in leaf_names(child)]

    def largest_unit(module):
        if atomic(module):
            return size(module)
        return max((largest_unit(child) for child in module[2]), default=0)

    devices = [*max_memory, "disk"]
    budget = dict(max_memory)
    if max_memory:
        budget[devices[0]] -= max((largest_unit(m) for m in modules), default=0)

    by_name = {module[0]: module for module in modules}
    queue = [module for module in modules if module[0] not in tied]
    device_map = {}
    current, used = 0, 0
    while queue:
        module = queue.pop(0)
        device = devices[current]
        inside = set(leaf_names(module))
        followers = [by_name[t] for t in tied if tied[t] in inside]
        needed = size(module) + sum(size(f) for f in followers)
        if device == "disk" or used + needed <= budget[device]:
            device_map[module[0]] = device
            for follower in followers:
                device_map[follower[0]] = device
            used += needed
        elif not atomic(module):
            queue[:0] = module[2]
        else:
            current, used = current + 1, 0
            queue.insert(0, module)
    return device_map
```

The loop always terminates because every iteration places a module, opens a container into its children, or advances to the next device, and `"disk"` accepts everything.

## Common mistakes

- **Packing instead of scanning.** First-fit or best-fit across all devices respects memory budgets but scatters consecutive layers across GPUs. Once a device is closed, it must stay closed.
- **Looking ahead.** Skipping a module that does not fit to place a smaller later module breaks forward order within a device.
- **Dividing a no-split module.** The check for opening a module must inspect its class against `no_split`, not just check whether it has children.
- **Never opening containers.** Treating every top-level module as indivisible pushes an entire layer stack onto the next device as soon as it does not fit whole.
- **Opening everything upfront.** Flattening the tree before scanning assigns the same devices to leaves but produces thousands of entries instead of naming containers that fit whole.
- **Mishandling the reserve.** Common bugs include omitting the reserve, subtracting it from every device instead of only the first, or computing it over top-level containers (which makes the whole layer stack the reserve and leaves the first device empty).
- **Using `<` instead of `<=`.** A module whose size equals the remaining budget fits.
- **Mishandling tied modules.** Placing tied modules independently or forgetting to include their extra bytes in `need` violates the placement rules.
- **Forgetting `"disk"`.** When the last listed device fills up, remaining modules spill to `"disk"` rather than overflowing the last device.

## Complexity

With $n$ modules and $d$ devices, each module is popped at most once per device it is tried on, and each device closes at most once, giving $O(n + d)$ queue operations. The reference solution recomputes module sizes and leaf lists on each visit, taking $O(n \cdot \text{depth})$ time in total for sizes. That is fast at this problem's scale, though caching them is worthwhile for models with hundreds of thousands of modules.

## In the wild

`accelerate.infer_auto_device_map` in Hugging Face Accelerate (`src/accelerate/utils/modeling.py`) powers `device_map="auto"` and `device_map="sequential"` in `transformers` using this same scan. Its documentation specifies the same device order (GPUs, then CPU, then memory-mapped disk), requires `no_split_module_classes` for blocks with residual connections, and assigns devices sequentially so that a first layer larger than GPU 0 sends the whole model to CPU or disk.

This problem simplifies several details of Accelerate's implementation:

- Accelerate inspects `torch.nn.Module` trees where containers can hold their own parameters and buffers, computing sizes from parameter shapes and dtypes.
- Accelerate recomputes the reserve dynamically from the largest *unplaced* module and applies it to both the first GPU and CPU (since CPU executes disk-offloaded layers when no GPU is present). Here, the reserve is fixed and applied only to the first device.
- Accelerate detects tied parameters by tensor identity, supports tied groups with more than two members, and tries opening a tied module if its owner fits alone but not with its followers.
- `"auto"` and `"balanced"` first shrink each GPU's memory limit via `get_balanced_memory` to spread layers evenly across GPUs before running the scan, whereas `"sequential"` uses the limits as given.
- Accelerate also supports a `fallback_allocation` mode that lets an empty device accept a later module when the next one does not fit, and cleans the final map by collapsing parents whose children all share one device.

At runtime, Accelerate attaches forward hooks that move each module's inputs to its assigned device and temporarily copy `"cpu"` or `"disk"` weights onto the execution device for the duration of the call. This is the exact staging the reserve accounts for.
