A safetensors file starts with an 8-byte length, then a JSON header that maps each tensor name to its dtype, shape, and byte range. Nothing forces a loader to read the rest: to load three tensors out of a thousand, read three ranges.

Write `load_tensors(file, names)`. `file` is an open binary file that counts the bytes read from it. Return a dictionary from each requested name to its raw bytes, reading no more than the header and the requested ranges.

For a header of $H$ bytes and requested tensors of sizes $n_1, \dots, n_k$, the budget is

$$
8 + H + \sum_{j=1}^{k} n_j
$$

## What is judged

| case | needs the model | limit on bytes read |
|---|---|---|
| a file written in the test | no | $8 + H + \sum n_j$ |
| two tensors of the Granite 4.0 1B checkpoint | yes | the same budget, on 1.6 GB |

```python
def load_tensors(file, names):
    """Return {name: bytes} reading only the header and the named tensors."""
```
