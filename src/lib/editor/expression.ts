const NAME_CHAR = /[A-Za-z0-9_]/;
const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

/**
 * The dotted name under a position in a line, up to and including the segment
 * the position is in: on `softmax` in `F.softmax(x)` it is `F.softmax`, on `F`
 * it is `F`. Null when the position is not on a name, or the name hangs off
 * something that is not a plain name (`f(x).shape`, `rows[0].size`): only
 * plain dotted names can be looked up without running the code.
 */
export function dottedNameAt(line: string, offset: number): string | null {
  let start = offset;
  let end = offset;
  while (start > 0 && NAME_CHAR.test(line[start - 1] ?? "")) start -= 1;
  while (end < line.length && NAME_CHAR.test(line[end] ?? "")) end += 1;
  if (start === end || !NAME.test(line.slice(start, end))) return null;
  const receiver = receiverBefore(line.slice(0, start));
  if (receiver === null) return null;
  return receiver === "" ? line.slice(start, end) : `${receiver}.${line.slice(start, end)}`;
}

/**
 * What a name being typed hangs off, given the line up to where the name
 * starts: `torch.nn` for `torch.nn.|`, the empty string for a bare name, and
 * null when it follows a dot that is not part of a plain dotted name.
 */
export function receiverBefore(before: string): string | null {
  if (!before.endsWith(".")) return "";
  const segments: string[] = [];
  let end = before.length;
  while (end > 0 && before[end - 1] === ".") {
    let start = end - 1;
    while (start > 0 && NAME_CHAR.test(before[start - 1] ?? "")) start -= 1;
    const segment = before.slice(start, end - 1);
    if (!NAME.test(segment)) return null;
    segments.unshift(segment);
    end = start;
  }
  // `f(x).shape`, `rows[0].size`, `"a".join`, `1.5.real`: the head is an expression, not a name.
  const previous = before[end - 1];
  if (previous !== undefined && /[)\]}"'\w]/.test(previous)) return null;
  return segments.join(".");
}
