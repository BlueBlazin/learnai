import { describe, expect, it } from "vitest";
import { findSteps, moveStep, type StepNode, stepNumber } from "./figure-steps";

const node = (localName: string, id = "", children: StepNode[] = [], textContent = ""): StepNode => ({
  localName,
  id,
  textContent,
  children,
});
const desc = (text: string) => node("desc", "", [], text);

describe("stepped figures", () => {
  it("reads the step number behind the sanitizer's id prefix", () => {
    expect(stepNumber("step-1")).toBe(1);
    expect(stepNumber("figr1-step-12", "figr1-")).toBe(12);
    expect(stepNumber("step-2", "figr1-")).toBeNull();
    expect(stepNumber("figr1-step-0", "figr1-")).toBeNull();
    expect(stepNumber("figr1-step-02", "figr1-")).toBeNull();
    expect(stepNumber("figr1-step-2b", "figr1-")).toBeNull();
    expect(stepNumber("figr1-steps", "figr1-")).toBeNull();
  });

  it("finds the frames in number order with their notes on one line", () => {
    const second = node("g", "f-step-2", [node("rect"), desc("  One new token.\n    The row is longer. ")]);
    const first = node("g", "f-step-1", [desc("Nothing is cached."), node("rect")]);
    const shared = node("g", "f-axes", [desc("Not a step.")]);
    const steps = findSteps({ children: [shared, second, node("rect"), first] }, "f-");
    expect(steps.map((step) => step.element)).toEqual([first, second]);
    expect(steps.map((step) => step.note)).toEqual([
      "Nothing is cached.",
      "One new token. The row is longer.",
    ]);
  });

  it("is an ordinary figure without step-1, and stops at the first gap", () => {
    expect(findSteps({ children: [node("g", "f-step-2"), node("g", "f-legend")] }, "f-")).toEqual([]);
    expect(findSteps({ children: [node("g", "step-1")] }, "f-")).toEqual([]);
    const steps = findSteps(
      { children: [node("g", "f-step-1"), node("g", "f-step-2"), node("g", "f-step-4")] },
      "f-",
    );
    expect(steps).toHaveLength(2);
  });

  it("counts only groups directly under the root, and gives a missing note as empty", () => {
    const nested = node("g", "f-wrap", [node("g", "f-step-1", [desc("Nested.")])]);
    expect(findSteps({ children: [nested] }, "f-")).toEqual([]);
    expect(findSteps({ children: [node("rect", "f-step-1")] }, "f-")).toEqual([]);
    const bare = node("g", "f-step-1", [node("g", "", [desc("A desc deeper down is not the note.")])]);
    expect(findSteps({ children: [bare] }, "f-")).toEqual([{ element: bare, note: "" }]);
  });

  it("moves between steps without leaving the range", () => {
    expect(moveStep(0, 1, 4)).toBe(1);
    expect(moveStep(3, 1, 4)).toBe(3);
    expect(moveStep(0, -1, 4)).toBe(0);
    expect(moveStep(2, -1, 4)).toBe(1);
  });
});
