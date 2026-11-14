Reshape `x` into groups of 64 once; both terms of the split sum reuse it.

---

The bias term needs only the per-group sum of `x`, which does not depend on the row.
