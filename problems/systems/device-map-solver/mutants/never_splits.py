# BUG: a module that does not fit is never opened: a container that is too large for the current device moves on whole and takes everything after it along.
def solve_device_map(modules, max_memory, no_split, tied):
    def size(module):
        _, _, body = module
        return body if isinstance(body, int) else sum(size(child) for child in body)

    def atomic(module):
        _, cls, body = module
        return isinstance(body, int) or cls in no_split

    def leaf_names(module):
        name, _, body = module
        if isinstance(body, int):
            return [name]
        return [leaf for child in body for leaf in leaf_names(child)]

    def largest_unit(module):
        # The largest piece that can never be divided: a leaf or a no-split module.
        if atomic(module):
            return size(module)
        return max((largest_unit(child) for child in module[2]), default=0)

    devices = [*max_memory, "disk"]
    budget = dict(max_memory)
    if max_memory:
        # Room on the first device for the inputs and outputs of whatever runs there.
        budget[devices[0]] -= max((largest_unit(m) for m in modules), default=0)

    tied_names = set(tied)
    by_name = {module[0]: module for module in modules}
    queue = [module for module in modules if module[0] not in tied_names]
    device_map = {}
    current, used = 0, 0
    while queue:
        module = queue.pop(0)
        device = devices[current]
        # Tied modules travel with the module that holds their owner.
        inside = set(leaf_names(module))
        followers = [by_name[t] for t in tied if tied[t] in inside]
        needed = size(module) + sum(size(f) for f in followers)
        if device == "disk" or used + needed <= budget[device]:
            device_map[module[0]] = device
            for follower in followers:
                device_map[follower[0]] = device
            used += needed
        else:
            current, used = current + 1, 0  # this device is closed for good
            queue.insert(0, module)
    return device_map
