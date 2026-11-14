How far back from the end can a pending sequence start? A character has at most four
bytes, and a pending one is missing at least one of them. And how do you recognize,
from a single byte, whether it is a lead byte or a continuation byte?

---

Walk backward from the end over at most three bytes, skipping continuation bytes
(`byte & 0xC0 == 0x80`). The first byte you meet that is not a continuation byte
decides everything: if it is a lead byte whose row of the table asks for more bytes
than you have seen from it to the end, the sequence may be pending; in every other
case (ASCII, a character with all its bytes, a byte that starts nothing, or three
continuation bytes in a row) nothing is pending and the answer is `len(data)`.

---

"May be pending" becomes "is pending" with one more check. The length comes from the
lead byte: `C2..DF` is 2, `E0..EF` is 3, `F0..F4` is 4, and everything else, including
`C0`, `C1`, and `F5..FF`, is never pending. If at least one byte follows the lead,
that second byte must be inside the lead's row: at least `A0` after `E0`, at most `9F`
after `ED`, at least `90` after `F0`, at most `8F` after `F4`. Third bytes need no
check: any continuation byte is allowed there, and you only skipped continuation bytes
on the way back.
