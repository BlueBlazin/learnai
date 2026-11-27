The index needs to make fetching all vectors in list $j$ cheap. If the vectors are stored grouped by list, each list is a contiguous slice. What extra arrays do you need so you know where every slice starts and ends (including empty lists) and can map each stored position back to its original corpus index?

---

In `build_ivf`, compute the `(N, C)` squared distance table by broadcasting and take the `argmin` along the centroid axis so ties pick the lower centroid id. Then sort the vector indices by their assigned list using a **stable** sort to obtain `ids`. Count the members of every list (keeping `0` for empty lists) and prepend a `0` to their cumulative sum to build `offsets`.

In `search`, call `distance(q, centroids)`, run a stable ascending sort on the result, and take the first `nprobe` entries as the lists to probe.

---

Collect the storage positions of the probed lists in scan order, then gather and score them in one call:

```python
bounds = offsets.tolist()
positions = concatenate([arange(bounds[j], bounds[j + 1]) for j in probe])
distances = distance(q, vectors[positions])  # the candidates, and only them
best = stable_argsort(distances)[:k]  # shorter than k if there are few
return ids[positions[best]], distances[best]
```

An empty list contributes an empty range, and slicing with `[:k]` handles cases where `k` exceeds the number of candidates. The final line performs two index lookups. First, `best` indexes into the candidates and `positions` maps each candidate to its storage slot. Then `ids` maps that storage slot back to its corpus index.
