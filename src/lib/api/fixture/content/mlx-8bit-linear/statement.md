The large house model, **Granite 4.2 3B**, ships as an 8-bit MLX checkpoint: each linear layer stores `uint8` codes with one scale and one bias per group of 64 weights. Implement the forward pass of such a layer without ever materializing the full float matrix.

Write `quantized_linear(x, codes, scales, biases)`. A group of 64 consecutive weights $w$ in a row is stored as codes $q \in \{0, \dots, 255\}$ with

$$
w_i = s \cdot q_i + b
$$

so a row's dot product with $x$ splits into two cheap terms:

$$
\sum_i w_i x_i = s \sum_i q_i x_i + b \sum_i x_i
$$

## What is judged

| case | needs the model | checks |
|---|---|---|
| one group, hand-written codes | no | the arithmetic |
| two rows, three groups | no | grouping and shapes |
| `layers.0.mlp.fc1` of the checkpoint | yes | output against the reference |
| peak memory on the checkpoint layer | yes | stays under $1.25\times$ the codes |

The cases marked "yes" read weights from the downloaded checkpoint (3.9 GB). Until it is on disk they are skipped and the rest still run.

```python
def quantized_linear(x, codes, scales, biases):
    """x: (d_in,). codes: (d_out, d_in) uint8. scales, biases: (d_out, d_in // 64)."""
```
