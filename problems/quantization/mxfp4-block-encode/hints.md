Break the task into the scale calculation and the element rounding. After dividing a block by its scale $X$, every magnitude lies in $[0, 8)$ and must round to one of $0, 0.5, 1, 1.5, 2, 3, 4, 6$. Notice how the step size changes across that list. Where is the spacing $0.5$, where is it $1$, and where is it $2$? For the scale, you need $\lfloor \log_2 m \rfloor$ of a positive float32 number. Float32 already stores that integer directly in its bit representation. Where?

---

**For the scale**, a positive normal float32 is $2^{\text{biased} - 127} \cdot 1.f$, with the 8-bit biased exponent stored in bits 23 through 30. Reinterpret the block maximum as an int32 (`view`) and shift right by 23 to get $\lfloor \log_2 m \rfloor + 127$ exactly, without any logarithm. Subtract 2 to obtain the E8M0 scale byte. To recover $X = 2^{\text{byte} - 127}$ as an exact float32, shift the int32 byte left by 23 and reinterpret as float32. (`frexp` and `ldexp` are equally exact where available. Float32 `floor(log2(m))` fails one ulp below a large power of two, and `pow(2, e)` is not always an exact power of two.) Handle all-zero blocks with `where`.

**For the elements**, clamp $a = |x| / X$ to at most $6$. Then round in units of the local step size: $\operatorname{round}(2a)$ for $a < 2$, $\operatorname{round}(a)$ for $2 \le a < 4$, and $\operatorname{round}(a / 2)$ for $4 \le a \le 6$.

---

Adding a region offset turns those rounded step counts directly into the 3-bit magnitude code:

$$
2E + M = \begin{cases}
\operatorname{round}(2a) & a < 2, \\
\operatorname{round}(a) + 2 & 2 \le a < 4, \\
\operatorname{round}(a / 2) + 4 & 4 \le a \le 6 .
\end{cases}
$$

Your framework's `round` uses round-half-to-even, and because every offset is even, an even count always corresponds to a code with mantissa bit 0. Values that round up across a region boundary still land on the right code (for instance, $a = 1.9$ gives $\operatorname{round}(3.8) = 4$, the code for $2.0$). Add 8 wherever $x < 0$ and cast to uint8.

To decode, widen `codes` and `exps` to int32 first. Look up the magnitude of `code & 7` in the 8-entry table `(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)`, compute the sign as `1 - 2 * (code >> 3)`, and multiply by each block's scale repeated 32 times along the row.
