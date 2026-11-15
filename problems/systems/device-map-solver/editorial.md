## The idea

A device map is a partition of the model's weights into consecutive runs, one per
device. "Consecutive" is the important word. During a forward pass the hidden state
is moved to wherever the next module lives, so the number of device-to-device
copies per pass is the number of boundaries in the map. Filling devices strictly in
order gives the minimum, one boundary per device used, and it turns placement into
a single scan: there is one current device, and the only question for the next
module is whether it still fits there.

That scan is greedy and does not maximize what the GPUs hold. A bin-packing
solution that tucks a small late layer into the free tail of GPU 0 holds more on the
GPUs and is slower, because the activations then visit GPU 0 twice. Giving up a few
hundred megabytes at the end of each device is the price of the order.

**What may be divided.** The scan works on a tree. A module that fits is placed
whole, which keeps the map short. A module that does not fit is opened and its
children are tried on the same device, so a stack of layers is cut between two
layers. The recursion stops at leaves and at modules of a no-split class. A
transformer block computes `x + f(x)`: if the block were cut, `x` would be on the
device where the block started and `f(x)` on the device where it ended, and the
addition fails. Hence every class with a residual connection is no-split.

**The reserve.** Weights are not all a device must hold. The first device also
receives the inputs, holds the output while it is post-processed, and executes
every module that was offloaded to CPU or disk: those weights are copied in just
before their forward pass and dropped after it. So the largest module that might
ever be brought in has to fit next to what lives there permanently. Reserving the
size of the largest indivisible module guarantees that.

**Tied weights.** An output head tied to the input embedding is one tensor with two
names. If the two names were placed independently, the head, which is last in
forward order, would land on the last device, the embedding on the first, and
loading would either duplicate the tensor or break the tie. The tied module
therefore travels with its owner: it is removed from the scan, its extra bytes are
added to the owner's when checking the fit, and both get the same device. This is
the one place where the map is allowed to go "backwards": the head on GPU 0 is used
after the last layer on GPU 1.

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

The loop terminates because every iteration places a module, replaces a container
by its children, or advances the device, and `"disk"` accepts everything.

## Common mistakes

- **Packing instead of scanning.** First-fit or best-fit over all devices ("put it
  wherever there is room") respects every budget and scatters consecutive layers.
  A closed device stays closed.
- **Looking ahead.** Skipping a module that does not fit to place a smaller later one
  is the same mistake inside one device.
- **Dividing a no-split module.** The check for "can be opened" must look at the
  class, not only at whether the module has children.
- **Never opening.** Treating every top-level module as indivisible sends the whole
  layer stack to the next device as soon as it does not fit as one piece.
- **Opening everything first.** Flattening the tree before the scan gives the same
  devices and a map with thousands of entries where one would do; the specified
  map names a module whole when it fits whole.
- **Forgetting the reserve**, or subtracting it from every device, or computing it
  from the top-level modules (the "largest layer" is then the whole stack and the
  first device stays empty).
- **`<` for `<=`.** An exact fit fits.
- **Tied modules placed on their own**, or following the owner without their bytes
  being counted.
- **No disk.** When the last listed device is full, what remains goes to `"disk"`,
  not onto the last device over its budget.

## Complexity

With $n$ modules, each is taken from the queue at most once per device it is tried
on, and a device is closed at most once: $O(n + d)$ queue operations for $d$
devices. The reference recomputes sizes and leaf lists per visit, which is
$O(n \cdot \text{depth})$ in total for sizes and fine at this scale; cache them for
models with hundreds of thousands of modules.

## In the wild

`accelerate.infer_auto_device_map` (Hugging Face Accelerate,
`src/accelerate/utils/modeling.py`), which `device_map="auto"` and
`device_map="sequential"` in `transformers` end in, is this scan. Its documentation
("Loading big models into memory") states the order (GPUs, then CPU, then disk as
memory-mapped tensors), that `no_split_module_classes` should list every block with
a residual connection, and that devices are attributed sequentially, "so if your
first layer is bigger than the size of the GPU you have, it will end up with
everything on the CPU/Disk". Differences from this problem, all simplifications
made here:

- Accelerate works on `torch.nn.Module`s whose containers can hold parameters and
  buffers of their own, and sizes come from parameter shapes and dtypes.
- Its reserve is the largest layer among the modules *not yet placed*, recomputed as
  the scan proceeds, and it is applied to the first GPU and to the CPU (the "main
  devices"), since the CPU executes what was offloaded to disk when there is no GPU.
  Here it is one fixed number on the first device.
- Tied parameters are found by identity of the tensors, a tied group can have more
  than two members, and when the owner fits alone but not with its tied modules,
  Accelerate tries to open the tied module before giving up on the device.
- `"auto"` and `"balanced"` first lower each GPU's limit so that the model spreads
  evenly over the GPUs (`get_balanced_memory`), then run the same scan;
  `"sequential"` runs it on the limits as given.
- A `fallback_allocation` option lets an empty device take a later module when the
  next one in order does not fit, trading order for use of the device, and the
  result is cleaned so that a parent whose children all share a device is reported
  once.

At run time, Accelerate's hooks move each module's input to the module's device,
and for modules mapped to `"cpu"` or `"disk"` load the weights onto the execution
device for the duration of the call: the mechanism the reserve leaves room for.
