Two separate problems. Inside a block, after dividing by the scale, every magnitude
lies in $[0, 8)$ and has to land on one of $0, 0.5, 1, 1.5, 2, 3, 4, 6$. Look at the
spacing of that list: where is the step $0.5$, where $1$, where $2$? And for the
scale: you need $\lfloor \log_2 m \rfloor$ of a float32 number. A float32 number
already stores exactly that. Where?

---

The scale. A positive float32 is $2^{E - 127} \cdot 1.f$, and $E$ sits in bits 23 to
30. Reinterpret the block maximum as an int32 (`view`) and shift right by 23: that is
$\lfloor \log_2 m \rfloor + 127$, exactly, with no logarithm. Subtract 2 and you have
the scale byte. In the other direction, an int32 holding `byte << 23`, reinterpreted
as float32, is exactly $2^{\text{byte} - 127}$. (`frexp` and `ldexp` do the same where
they exist. `floor(log2(m))` computed in float32 is wrong one ulp below a large power
of two, and a float32 `pow(2, e)` is not always a power of two.) Handle the all-zero
block with a `where`.

The elements. Clamp $a = |x| / X$ to at most $6$. Then round in units of the local
step: $\operatorname{round}(2a)$ where $a < 2$, $\operatorname{round}(a)$ where
$2 \le a < 4$, $\operatorname{round}(a / 2)$ above.

---

The rounded counts are the codes, up to an offset per region:

$$
2E + M = \begin{cases}
\operatorname{round}(2a) & a < 2, \\
\operatorname{round}(a) + 2 & 2 \le a < 4, \\
\operatorname{round}(a / 2) + 4 & 4 \le a \le 6 .
\end{cases}
$$

Your framework's `round` rounds half to even, and an even count is a code whose
mantissa bit is 0, so the ties come out as specified. A value that rounds up across a
region boundary still gets the right code: $a = 1.9$ gives
$\operatorname{round}(3.8) = 4$, the code of $2.0$. Add 8 where $x < 0$ and cast to
uint8.

To decode, widen the codes and the scale bytes to int32 first. The magnitude is a
lookup of `code & 7` in the 8-entry table, the sign is `1 - 2 * (code >> 3)`, and the
scale of each block is repeated 32 times along the row.
