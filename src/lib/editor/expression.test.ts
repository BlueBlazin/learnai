import { describe, expect, it } from "vitest";
import { completionSortKey, completionValidity, kindWord, markupText, SERVER_LIST_LIMIT } from "./completion";
import { dottedNameAt, receiverBefore } from "./expression";

describe("dottedNameAt", () => {
  const line = "    y = F.scaled_dot_product_attention(q, torch.linalg.svd(x).S)";
  const on = (text: string, offset = 0) => dottedNameAt(line, line.indexOf(text) + offset);

  it("returns the name up to the segment under the position", () => {
    expect(on("scaled", 3)).toBe("F.scaled_dot_product_attention");
    expect(on("F.")).toBe("F");
    expect(on("svd", 1)).toBe("torch.linalg.svd");
    expect(on("linalg")).toBe("torch.linalg");
    expect(on("y =")).toBe("y");
  });

  it("is null off a name, and for names that hang off an expression", () => {
    expect(on(" = ", 2)).toBeNull();
    expect(on(".S)", 1)).toBeNull();
    expect(dottedNameAt("rows[0].size", 9)).toBeNull();
    expect(dottedNameAt("x = 12", 5)).toBeNull();
  });
});

describe("receiverBefore", () => {
  it("names what a member is being completed on", () => {
    expect(receiverBefore("    y = torch.")).toBe("torch");
    expect(receiverBefore("torch.nn.functional.")).toBe("torch.nn.functional");
    expect(receiverBefore("    y = ")).toBe("");
    expect(receiverBefore("")).toBe("");
  });

  it("is null after a dot that does not follow a plain name", () => {
    expect(receiverBefore("f(x).")).toBeNull();
    expect(receiverBefore("rows[0].")).toBeNull();
    expect(receiverBefore('"a".')).toBeNull();
    expect(receiverBefore("f(x).shape.")).toBeNull();
  });
});

describe("completion helpers", () => {
  it("orders what matches equally: public names first, a prefix before its extensions", () => {
    const labels = ["export", "__exp__", "expm1", "_exp", "exp", "exp2", "Exp"];
    const sorted = [...labels].sort((a, b) => completionSortKey(a).localeCompare(completionSortKey(b)));
    expect(sorted.slice(0, 2).map((label) => label.toLowerCase())).toEqual(["exp", "exp"]);
    expect(sorted.indexOf("exp")).toBeLessThan(sorted.indexOf("export"));
    expect(sorted.indexOf("exp")).toBeLessThan(sorted.indexOf("exp2"));
    expect(sorted.slice(-2)).toEqual(["_exp", "__exp__"]);
  });

  it("names kinds in words", () => {
    expect(kindWord(3)).toBe("function");
    expect(kindWord(9)).toBe("module");
    expect(kindWord(7)).toBe("class");
    expect(kindWord(undefined)).toBe("");
    expect(kindWord(999)).toBe("");
  });

  it("reads documentation in every shape the protocol allows", () => {
    expect(markupText("plain")).toBe("plain");
    expect(markupText({ kind: "markdown", value: "**bold**" })).toBe("**bold**");
    expect(markupText({ language: "python", value: "x = 1" })).toBe("```python\nx = 1\n```");
    expect(markupText(["a", { kind: "plaintext", value: "b" }])).toBe("a\n\nb");
    expect(markupText(undefined)).toBeNull();
    expect(markupText({})).toBeNull();
  });

  it("narrows a list in the editor while the word grows, and asks again otherwise", () => {
    const check = (validity: ReturnType<typeof completionValidity>, text: string) =>
      validity === undefined ? "ask" : typeof validity === "function" ? validity(text) : validity.test(text);
    // The server's list for `torch.ex`: everything that matches "ex", marked incomplete.
    const narrowed = completionValidity("ex", true, 88);
    expect(check(narrowed, "exp")).toBe(true);
    expect(check(narrowed, "ex")).toBe(true);
    expect(check(narrowed, "e")).toBe(false);
    expect(check(narrowed, "ex.")).toBe(false);
    // A list that hit the server's limit may be missing names: always ask again.
    expect(check(completionValidity("", true, SERVER_LIST_LIMIT), "e")).toBe("ask");
    // A complete list is good for any word.
    expect(check(completionValidity("ex", false, 12), "e")).toBe(true);
    expect(check(completionValidity("ex", false, 12), "e(")).toBe(false);
  });
});
