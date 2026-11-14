Look at every number in the formulas that is not `x` or a parameter: $\epsilon$,
$\tfrac{1}{2}$, $1$, $\sqrt{2/\pi}$, $0.044715$. For each, what Python object will
carry it into the graph, and what dtype does MLX give the result when that object
meets a float16 array?

---

Keep constants as Python floats (`1e-5`, `math.sqrt(2.0 / math.pi)`): MLX treats a
Python scalar as weakly typed and converts it to the array's dtype. Wrapping one in
`mx.array(...)` makes a float32 array, and float16 combined with float32 is float32.
While you write the block, check `.dtype` after each line (reading a dtype evaluates
nothing). Do not fix a float32 result with `.astype(mx.float16)` at the end: the
activations in between were already computed in float32.

---

```python
rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
h = (x / rms * params["gain"]) @ params["w_in"].T
h = 0.5 * h * (1 + mx.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * h * h * h)))
```

The epsilon must be a float16 number greater than zero: `1e-5` is stored as
`1.0014e-05`. Float16 has no value between 0 and about `6e-8`, so the `1e-8` of
float32 code becomes exactly 0 and a zero row divides by zero.
