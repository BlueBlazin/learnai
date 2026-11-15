## The idea

A Triton kernel is the body of a loop you do not write. The launch
`rmsnorm_kernel[(n_rows,)](...)` runs it once per grid point, concurrently, and the
only thing that distinguishes one run from another is `tl.program_id(0)`. Inside,
there are no tensors with shapes, only pointers and blocks: a block is a small
fixed-size vector that lives in registers, and `tl.load` and `tl.store` move blocks
between registers and memory.

**One program per row** is the natural decomposition for a normalization: the row's
sum of squares is needed by every element of the row and by nothing else, so a
program computes it once and uses it `n_cols` times. Finer (one program per element)
recomputes the sum `n_cols` times; coarser would leave the GPU's parallelism unused.

**Blocks have a fixed size; rows do not.** `tl.arange(0, BLOCK)` needs a
compile-time power of two, so a row of 37 columns in blocks of 16 is three blocks
covering 48 columns. The last eleven do not exist. For row 0 they are the first
eleven elements of row 1; for the last row they are whatever the allocator placed
after the tensor. The mask `cols < n_cols` is the only thing standing between the
kernel and that memory:

- on a load, masked-off elements are not read and take the value of `other`. For a
  sum the neutral value is 0. Without `other` they are undefined, which in practice
  means stale register contents;
- on a store, masked-off elements are not written.

**Two passes.** The root mean square must be known before the first element can be
written, and a block cannot hold a whole row, so the row is read twice: once to
accumulate, once to normalize. The accumulator is a block of `BLOCK` partial sums,
reduced with `tl.sum` after the loop, which keeps the loop body a plain vector
add.

**float32 inside.** In half precision the largest finite value is 65504, so a
square overflows once $|x| > 255$. `x.to(tl.float32)` before squaring costs nothing
(the conversion happens in registers) and the store converts the result back. The
divisor is `n_cols`, the number of real columns, not the number of columns the
blocks covered, and `eps` goes inside the root, as in `rmsnorm`.

## Reference solution (PyTorch)

```python
@tl.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * n_cols
    out_row = out_ptr + row * n_cols

    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        acc += x * x
    rms = tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)

    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_row + cols, x / rms * w, mask=mask)


def rms_norm(x, weight, eps=1e-5, block=64):
    n_cols = x.shape[-1]
    rows = x.reshape(-1, n_cols).contiguous()
    out = torch.empty_like(rows)
    rmsnorm_kernel[(rows.shape[0],)](rows, weight.contiguous(), out, n_cols, eps, BLOCK=block)
    return out.reshape(x.shape)
```

The wrapper's `.contiguous()` is not a formality. A kernel receives the address of
the tensor's first element and nothing else; `x[:, ::2]` has the right shape and its
elements are two apart in memory. PyTorch operations honor strides; a kernel that
computes `row * n_cols + col` does not. (`.contiguous()` returns the tensor itself
when the layout is already right, so the common case copies nothing.) The output
needs no initialization, since the kernel writes every element.

On a real GPU the block size is a tuning knob: larger blocks mean fewer loop
iterations and wider vector operations per program. Triton's own layer-norm tutorial
picks the next power of two above the row length, capped so that a block fits in
64 KiB. The masked tail is what makes any choice correct.

## Common mistakes

- **No mask.** Row 0's sum includes elements of row 1 and its store overwrites
  them; the last row goes past the end of the tensor. The shim raises
  `tl.OutOfBounds` or `tl.DataRace`; a GPU raises nothing.
- **Masked loads but an unmasked store.** The statistics are right and each program
  scribbles over the start of the next row, racing with the program that owns it.
- **`cols <= n_cols`.** One element too many, on every row.
- **`other=1.0`** (or any value that is not neutral for the reduction). Each missing
  column adds 1 to the sum of squares. Rows whose length is a multiple of the block
  are unaffected, which is how this survives a test with round sizes.
- **Dividing by the padded length.** The mean is over `n_cols` columns.
- **eps outside the root.** Indistinguishable on activations of order 1, wrong on
  small ones.
- **No conversion to float32.** Infinite squares for float16 activations above 255,
  and a row of zeros.
- **One block and no loop.** Triton's tutorial does this on purpose, with a block as
  long as the row; with a fixed `BLOCK` everything after the first block is ignored.
- **One program per element.** Correct, with `n_cols` times the reads.
- **A strided tensor passed to the kernel**, or an output allocated as float32
  whatever the input is.

## Complexity

`n_rows` programs, each making two passes over its row: $2 \cdot \text{n\_cols}$
elements of `x` and $\text{n\_cols}$ gains loaded, $\text{n\_cols}$ elements stored,
in $2 \lceil \text{n\_cols} / \text{BLOCK} \rceil$ loop iterations. No memory beyond
the output.

## In the wild

The forward kernel of the layer-norm tutorial in Triton's documentation has this
exact structure (a pointer advanced by `row * stride`, `for off in range(0, N,
BLOCK_SIZE)`, `tl.load(..., mask=cols < N, other=0.)`, a float32 accumulator reduced
with `tl.sum`), with a mean subtraction added. The kernels `torch.compile` generates
for a normalization look the same after fusion.

What was verified where. On the CPU, under the kit's `tritonshim`: everything the
judge checks. The shim was written from Triton's documentation and has never been
compared with Triton itself, which does not install on the machine the problem was
written on. Three things in particular rest on the shim author's reading rather
than on a documented statement: that Python scalars passed at launch arrive as
int32 and float32, that an integer divided by an integer with `/` gives float32,
and that every block shape must be a power of two. The reference kernel does not
depend on the second, and the tutorial depends on the first and third in the same
way the reference does. One known difference: the tutorial writes
`tl.zeros([BLOCK_SIZE], dtype=tl.float32)` with a list, which the shim refuses; a
tuple, as documented for `tl.zeros`, works in both. The GPU cases, which hand this
kernel to the real compiler, have never been run.
