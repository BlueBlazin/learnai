`mx.get_peak_memory()` and `mx.get_active_memory()` report process-wide totals since startup. To isolate what a single call used, what must you run and record *before* calling `fn`, and what must finish *after* calling `fn` before you read the counters?

---

Before the call, evaluate the arguments so their cost is already paid, synchronize, clear the buffer cache, reset the peak counter, and record active memory as a baseline. After the call, evaluate the result, wait for the GPU with `mx.synchronize()`, and subtract the baseline from both peak and active memory. Read the cache memory directly since it started empty.

`mx.eval` accepts arbitrarily nested tuples, lists, and dicts and ignores non-arrays, so `mx.eval(args)` and `mx.eval(result)` work for any inputs and outputs of `fn`.

---

```python
mx.eval(args)
mx.synchronize()
mx.clear_cache()
mx.reset_peak_memory()
baseline = mx.get_active_memory()

result = fn(*args)
mx.eval(result)
mx.synchronize()
```

Compute `peak_bytes = mx.get_peak_memory() - baseline` and `active_bytes = mx.get_active_memory() - baseline`, and read `mx.get_cache_memory()` directly. For `predicted_bytes`, use `math.prod(shape)` for the element count (which returns `1` for `()`) and `dtype.size` for the byte width of each element.
