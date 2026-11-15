The index must make one operation cheap: "give me all vectors of list $j$". If the
vectors are stored grouped by list, that is a slice. What do you need to store so
that you know where each slice begins and ends, including for a list with no
members, and so that a position in the grouped storage can be turned back into a
corpus index?

---

`build_ivf`: compute the `(N, C)` table of squared distances by broadcasting, take
the argmin along the centroid axis (first minimum: the lower id), and sort the
vector indices by their list number with a **stable** sort; that is `ids`. Count
the members of every list, with a count of 0 for lists that have none, and prefix
the cumulative sum with a 0; that is `offsets`.

`search`: one `distance(q, centroids)` call, a stable ascending sort of the result,
and the first `nprobe` entries are the lists to probe.

---

Collect the *positions* of the probed lists in scan order, then gather once:

```python
bounds = offsets.tolist()
positions = concatenate([arange(bounds[j], bounds[j + 1]) for j in probe])
distances = distance(q, vectors[positions])  # the candidates, and only them
best = stable_argsort(distances)[:k]  # shorter than k if there are few
return ids[positions[best]], distances[best]
```

An empty list contributes an empty range, and slicing with `[:k]` handles `k`
larger than the number of candidates. The last line has two lookups: `best` indexes
candidates, `positions` turns a candidate into a storage position, and `ids` turns
that into a corpus index.
