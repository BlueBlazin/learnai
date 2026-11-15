Two questions. First, the table: evenly spaced *probabilities*, pushed through the
inverse normal distribution function, give levels that are unevenly spaced in *value*.
If you used the same number of probabilities on both sides, would any level be zero?
Second, the quantizer: once a block has been divided by its largest magnitude, every
value lies in $[-1, 1]$. The codes are not positions on a ruler. How do you find the
closest entry of a 16-entry table for every element at once?

---

For the table, build the positive half from 8 probabilities and the negative half from
7, `linspace(0.9677083, 0.5, 9)` and `linspace(0.9677083, 0.5, 8)` without their last
element (the last element is $0.5$, whose quantile is $0$). The inverse normal
distribution function is `torch.special.ndtri` and `jax.scipy.special.ndtri`; in MLX,
$\Phi^{-1}(p) = \sqrt{2}\,\operatorname{erfinv}(2p - 1)$. Concatenate the negated
negative half, a single $0$, and the positive half in ascending order, and divide by
the maximum.

For the quantizer, reshape `w` to `(out, in // block_size, block_size)` as in
`groupwise-4bit`, take the maximum of the absolute values over the last axis, and
replace a zero maximum by 1 before dividing.

---

Add a trailing axis to the normalized blocks and subtract the table: the result has
shape `(out, n_blocks, block_size, 16)`, and

$$
\text{codes} = \operatorname*{arg\,min}_{k}\ \big|\, \text{normalized}[\dots, \text{None}] - \text{levels}\, \big|
$$

over that last axis is the nearest level. Cast to uint8 and reshape to `(out, in)`.
Dequantizing is `levels[codes]` (cast the codes to the index type your framework
wants) times `absmax` repeated `block_size` times along the input axis, each value
`block_size` times in a row, not the whole row of scales tiled.
