`mx.get_peak_memory()` and `mx.get_active_memory()` describe the whole process, since
it started. To turn them into "what did this call use", what do you have to do (and
read) *before* the call, and what has to be true of the result before you read them
*after* it?

---

Before the call: evaluate the arguments (so their cost is already paid), empty the
buffer cache, reset the peak, and note the active memory as a baseline. After the
call: evaluate the result, wait for the GPU with `mx.synchronize()`, then read the
peak and the active memory and subtract the baseline from each. The cache is read
as it is; it was empty before the call.

`mx.eval` accepts any nesting of tuples, lists, and dicts and ignores what is not an
array, so `mx.eval(args)` and `mx.eval(result)` work whatever `fn` takes and returns.

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

Then `peak_bytes = mx.get_peak_memory() - baseline`, and the same for the active
memory. For `predicted_bytes`, `math.prod(shape)` is the number of elements (it is 1
for `()`), and `dtype.size` is the number of bytes per element.
