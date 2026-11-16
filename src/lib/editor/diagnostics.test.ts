import { describe, expect, it } from "vitest";
import { type ServerDiagnostic, severityOf, visibleDiagnostics } from "./diagnostics";

const at = (severity: number | undefined, message: string): ServerDiagnostic => ({
  range: { start: { line: 0, character: 0 }, end: { line: 0, character: 1 } },
  ...(severity === undefined ? {} : { severity }),
  message,
});

describe("visibleDiagnostics", () => {
  it("drops hints and keeps everything else in order", () => {
    const diagnostics = [
      at(1, "Module `torch` has no member `exxp`"),
      at(4, "`y` is unused"),
      at(2, "a warning"),
      at(4, "`z` is unused"),
      at(3, "a note"),
      at(undefined, "no severity means error"),
    ];
    expect(visibleDiagnostics(diagnostics).map((d) => d.message)).toEqual([
      "Module `torch` has no member `exxp`",
      "a warning",
      "a note",
      "no severity means error",
    ]);
  });

  it("maps severities to the editor's three", () => {
    expect(severityOf(at(1, ""))).toBe("error");
    expect(severityOf(at(undefined, ""))).toBe("error");
    expect(severityOf(at(2, ""))).toBe("warning");
    expect(severityOf(at(3, ""))).toBe("info");
  });
});
