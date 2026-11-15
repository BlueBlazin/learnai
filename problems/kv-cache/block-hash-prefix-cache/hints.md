Two prompts can contain the same block of tokens at the same position and still
need different keys and values for it. What else, besides the block's own tokens,
determines the keys and values of a block, and how can a key capture that without
hashing the whole prefix again for every block?

---

Chain the keys: start from a constant, and for each full block compute
`key = H(previous_key, block_tokens)`. Equal keys then mean equal prefixes, and a
prompt of $n$ tokens is keyed in $O(n)$. With the list of keys in hand, every
method is short:

- `lookup`: count leading keys that are in the cache.
- `acquire`: increment the counts of the hits **first**; then insert the remaining
  keys one by one, evicting from the front of the eviction order when the cache is
  full and stopping when there is nothing to evict.
- `release`: walk the request's keys in **reverse**, decrement, and append a key
  to the eviction order when its count reaches 0.

---

Three containers: `refs` (key to count; its size is `cached_blocks()`), `held`
(request id to its list of keys), and the eviction order, for which an
`OrderedDict` gives O(1) "append at the back", "pop from the front", and "remove
from the middle":

```
def keys(tokens):
    key, out = b"", []
    for each full block:  key = sha256(key + bytes of the block);  out.append(key)

acquire:
    hits = number of leading keys in refs
    for key in keys[:hits]:  refs[key] += 1;  evictable.pop(key, None)
    held = hits
    for key in keys[hits:]:
        if len(refs) == capacity:
            if not evictable: break
            del refs[evictable.popitem(last=False)[0]]
        refs[key] = 1;  held += 1
    self.held[request_id] = keys[:held]

release:
    for key in reversed(self.held.pop(request_id)):
        refs[key] -= 1
        if refs[key] == 0: evictable[key] = None
```

The partial block at the end never gets a key: loop over
`range(0, len(tokens) - block_size + 1, block_size)`.
