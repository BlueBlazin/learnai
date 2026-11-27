How far back from the end can a pending sequence start? A UTF-8 character has at most four bytes, and a pending sequence is missing at least one of them. Also, how can you tell from a single byte whether it is a lead byte or a continuation byte?

---

Walk backward from the end over at most three bytes, skipping continuation bytes (`byte & 0xC0 == 0x80`). The first non-continuation byte you find decides the answer. If it is a lead byte whose table row requires more bytes than remain from it to the end, the suffix may be pending. In every other case (ASCII, a complete character, an invalid byte, or three continuation bytes in a row), nothing is pending and the answer is `len(data)`.

---

One more check turns "may be pending" into "is pending." Map the lead byte to its expected length (`C2..DF` is 2, `E0..EF` is 3, and `F0..F4` is 4). All other bytes, including `C0`, `C1`, and `F5..FF`, are never pending. If at least one byte follows the lead byte, verify that the second byte falls in the lead byte's allowed range: at least `A0` after `E0`, at most `9F` after `ED`, at least `90` after `F0`, and at most `8F` after `F4`. Third bytes need no extra check because any continuation byte is valid there, and you only skipped continuation bytes while walking backward.
