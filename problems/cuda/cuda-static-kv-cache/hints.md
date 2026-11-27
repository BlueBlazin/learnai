If the buffers always have `max_len` slots, how does attention know which slots hold valid tokens? And if `pos` must stay on the device, which indexing operations are ruled out, and which PyTorch method writes into slots specified by a tensor of indices?

---

In `__init__`, allocate the key and value buffers with list comprehensions (one `torch.zeros` per layer, passing `device=` and `dtype=`), plus a slot index tensor `torch.arange(max_len, device=device)`. Since `T = k.shape[2]` is already a Python `int`, you can compute the target slots on the device as `pos + slots[:T]` and write in place along axis 2 with `index_copy_`. For `mask`, compare the row of all slot indices against a column giving the last visible slot for each new token. In `reset`, zero each buffer in place.

---

```python
index = pos + self._slots[: k.shape[2]]
self.k[layer].index_copy_(2, index, k)
self.v[layer].index_copy_(2, index, v)
return self.k[layer], self.v[layer]
```

And for the mask:

```python
last_visible = pos + self._slots[:n_new]  # (n_new,)
return self._slots[None, :] <= last_visible[:, None]  # (n_new, max_len)
```

Keep the trailing underscore on `index_copy_`. Without it, PyTorch returns a new tensor and leaves the buffer unchanged.
