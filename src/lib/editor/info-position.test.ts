import { describe, expect, it } from "vitest";
import { type Box, placeInfo } from "./info-position";

const space: Box = { top: 8, left: 8, bottom: 842, right: 1352 };
const box = (left: number, top: number, width: number, height: number): Box => ({
  left,
  top,
  right: left + width,
  bottom: top + height,
});

describe("placeInfo", () => {
  it("starts level with the top of a list that hangs below the cursor", () => {
    const list = box(300, 509, 274, 174);
    const placed = placeInfo({
      list,
      option: box(304, 513, 266, 20),
      info: box(0, 0, 400, 354),
      space,
      above: false,
    });
    expect(placed).toMatchObject({ side: "top", offset: 0, where: "right", maxWidth: 400 });
  });

  it("is no taller than the room under the top of the list, so it never covers the cursor line", () => {
    const list = box(720, 509, 274, 174);
    const placed = placeInfo({
      list,
      option: box(724, 653, 266, 20),
      info: box(0, 0, 400, 354),
      space,
      above: false,
    });
    expect(placed.offset).toBe(0);
    expect(placed.maxHeight).toBe(space.bottom - list.top);
    expect(placed.maxHeight).toBeLessThan(354);
  });

  it("goes to the left when the right has no room", () => {
    const list = box(1040, 300, 274, 174);
    const placed = placeInfo({
      list,
      option: box(1044, 304, 266, 20),
      info: box(0, 0, 400, 200),
      space,
      above: false,
    });
    expect(placed.where).toBe("left");
    expect(placed.maxWidth).toBe(400);
  });

  it("ends level with the bottom of a list that sits above the cursor", () => {
    const list = box(200, 500, 274, 174);
    const placed = placeInfo({
      list,
      option: box(204, 504, 266, 20),
      info: box(0, 0, 400, 354),
      space,
      above: true,
    });
    expect(placed).toMatchObject({ side: "bottom", offset: 0, where: "right" });
    expect(placed.maxHeight).toBe(list.bottom - space.top);
  });

  it("goes under the selected row when neither side has room", () => {
    const narrow: Box = { top: 8, left: 8, bottom: 700, right: 520 };
    const list = box(120, 200, 300, 174);
    const option = box(124, 224, 292, 20);
    const placed = placeInfo({
      list,
      option,
      info: box(0, 0, 400, 200),
      space: narrow,
      above: false,
    });
    expect(placed).toMatchObject({ side: "top", where: "right-narrow" });
    expect(placed.offset).toBe(option.bottom - list.top);
    expect(placed.maxWidth).toBe(narrow.right - list.left - 30);
    expect(placed.maxHeight).toBe(narrow.bottom - option.bottom);
  });

  it("goes over the selected row when there is more room above than below", () => {
    const narrow: Box = { top: 8, left: 8, bottom: 700, right: 520 };
    const list = box(120, 500, 300, 174);
    const option = box(124, 524, 292, 20);
    const placed = placeInfo({
      list,
      option,
      info: box(0, 0, 400, 200),
      space: narrow,
      above: true,
    });
    expect(placed).toMatchObject({ side: "bottom", where: "right-narrow" });
    expect(placed.offset).toBe(list.bottom - option.top);
    expect(placed.maxHeight).toBe(option.top - narrow.top);
  });
});
