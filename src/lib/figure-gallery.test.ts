import { describe, expect, it } from "vitest";
import type { ProblemSummary, Track } from "@/lib/api";
import { figureParagraphs, groupGallery } from "./figure-gallery";

describe("figureParagraphs", () => {
  it("takes the lines that are only an image", () => {
    const text = [
      "Opening paragraph with ![inline](assets/a.svg) in the middle.",
      "",
      "![Caption with $x[i]$ math.](assets/map.svg)",
      "",
      "![Ten tokens.](interactive:sampler-order)",
      "",
      "![Not alone.](assets/b.svg) trailing words",
    ].join("\n");
    expect(figureParagraphs(text)).toEqual([
      "![Caption with $x[i]$ math.](assets/map.svg)",
      "![Ten tokens.](interactive:sampler-order)",
    ]);
  });

  it("ignores an example inside a code fence", () => {
    const text = "```markdown\n![Example.](assets/x.svg)\n```\n\n![Real.](assets/y.svg)\n";
    expect(figureParagraphs(text)).toEqual(["![Real.](assets/y.svg)"]);
  });

  it("finds none in a text without figures", () => {
    expect(figureParagraphs("Plain text.\n")).toEqual([]);
  });
});

describe("groupGallery", () => {
  const track = (id: string, order: number) => ({ id, title: `Track ${id}`, order }) as Track;
  const problem = (id: string, order: number) => ({ id, title: `Problem ${id}`, order }) as ProblemSummary;

  it("groups by module in catalog order and drops problems without figures", () => {
    const groups = groupGallery(
      [
        { track: "b", id: "b2", figures: ["![x](assets/x.svg)"] },
        { track: "a", id: "a1", figures: ["![x](assets/x.svg)"] },
        { track: "b", id: "b1", figures: ["![x](assets/x.svg)"] },
        { track: "a", id: "a0", figures: [] },
      ],
      [problem("b1", 1), problem("b2", 2), problem("a1", 1)],
      [track("b", 1), track("a", 2)],
    );
    expect(groups.map((group) => [group.id, group.problems.map((p) => p.id)])).toEqual([
      ["b", ["b1", "b2"]],
      ["a", ["a1"]],
    ]);
    expect(groups[0]?.title).toBe("Track b");
    expect(groups[0]?.problems[0]?.title).toBe("Problem b1");
  });

  it("falls back to ids when the catalog does not know a problem or module", () => {
    const [group] = groupGallery([{ track: "z", id: "p", figures: ["![x](assets/x.svg)"] }], [], []);
    expect(group?.title).toBe("z");
    expect(group?.problems[0]?.title).toBe("p");
  });
});
