Two prompts can contain the same block of tokens at the same position and still need different KV entries for it. What else, besides the block's own tokens, determines the keys and values of a block? How can a block's key capture that information without hashing the entire prefix from scratch for every block?

---

Chain the block keys. Start from a constant, and for each full block compute `key = H(previous_key, block_tokens)`. Equal keys then imply equal prefixes, and keying a prompt of $n$ tokens takes $O(n)$ time. Once you have the list of keys, each method is straightforward:

- `lookup`: count how many leading keys are present in the cache.
- `acquire`: increment the reference counts of the hit keys **first** (and remove them from the eviction queue). Then insert the remaining keys one by one, evicting from the front of the eviction queue when the cache is full and stopping if the queue is empty.
- `release`: iterate over the request's held keys in **reverse**, decrement each count, and append a key to the back of the eviction queue when its count reaches 0.

---

Use three containers: `refs` (mapping each key to its reference count, so `len(refs)` is `cached_blocks()`), `held` (mapping each `request_id` to its list of keys), and an `OrderedDict` for `evictable`. An `OrderedDict` supports $O(1)$ append to the back, $O(1)$ pop from the front, and $O(1)$ removal from the middle:

```
def keys(tokens):
    key, out = b"", []
    for each full block:
        key = sha256(key + bytes of the block)
        out.append(key)
    return out

acquire:
    hits = number of leading keys in refs
    for key in keys[:hits]:
        refs[key] += 1
        evictable.pop(key, None)
    held = hits
    for key in keys[hits:]:
        if len(refs) == capacity:
            if not evictable:
                break
            del refs[evictable.popitem(last=False)[0]]
        refs[key] = 1
        held += 1
    self.held[request_id] = keys[:held]

release:
    for key in reversed(self.held.pop(request_id)):
        refs[key] -= 1
        if refs[key] == 0:
            evictable[key] = None
```

Ignore any trailing partial block by iterating over `range(0, len(tokens) - block_size + 1, block_size)`.
