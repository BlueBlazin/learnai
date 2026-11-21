import { describe, expect, it } from "vitest";
import { SVG_ELEMENTS, sanitizeAttribute, sanitizeStyle } from "./svg-policy";

const keep = (element: string, name: string, value: string) => sanitizeAttribute(element, name, value, "f1-");

describe("SVG figure policy", () => {
  it("allows shapes and text, and nothing that runs, loads, or embeds", () => {
    for (const element of ["svg", "g", "path", "circle", "text", "tspan", "marker", "defs", "use"]) {
      expect(SVG_ELEMENTS.has(element)).toBe(true);
    }
    for (const element of [
      "script",
      "style",
      "foreignObject",
      "image",
      "a",
      "animate",
      "set",
      "animateTransform",
      "iframe",
      "feImage",
      "filter",
      "switch",
    ]) {
      expect(SVG_ELEMENTS.has(element)).toBe(false);
    }
  });

  it("keeps geometry and paint attributes as written", () => {
    expect(keep("path", "d", "M0 0L10 10a5 5 0 0 1 -3.5,2z")).toBe("M0 0L10 10a5 5 0 0 1 -3.5,2z");
    expect(keep("svg", "viewBox", "0 0 640 300")).toBe("0 0 640 300");
    expect(keep("path", "stroke", "currentColor")).toBe("currentColor");
    expect(keep("g", "transform", "translate(10 20) rotate(-30)")).toBe("translate(10 20) rotate(-30)");
    expect(keep("text", "font-size", "13")).toBe("13");
    expect(keep("svg", "aria-label", "Rotation of a pair: before, and after.")).toBe(
      "Rotation of a pair: before, and after.",
    );
  });

  it("drops event handlers and attributes it does not know", () => {
    expect(keep("svg", "onload", "alert(1)")).toBeNull();
    expect(keep("path", "onclick", "alert(1)")).toBeNull();
    expect(keep("g", "class", "fixed inset-0")).toBeNull();
    expect(keep("g", "data-x", "1")).toBeNull();
    expect(keep("svg", "xmlns:xlink", "http://www.w3.org/1999/xlink")).toBeNull();
    expect(keep("text", "filter", "url(#blur)")).toBeNull();
  });

  it("allows references only to the same file, and prefixes them with the ids", () => {
    expect(keep("marker", "id", "arrow")).toBe("f1-arrow");
    expect(keep("path", "marker-end", "url(#arrow)")).toBe("url(#f1-arrow)");
    expect(keep("path", "fill", "url('#grad')")).toBe("url(#f1-grad)");
    expect(keep("use", "href", "#dot")).toBe("#f1-dot");
    expect(keep("use", "xlink:href", "#dot")).toBe("#f1-dot");

    expect(keep("use", "href", "https://example.com/x.svg#dot")).toBeNull();
    expect(keep("use", "href", "other.svg#dot")).toBeNull();
    expect(keep("use", "href", "javascript:alert(1)")).toBeNull();
    expect(keep("use", "href", "data:image/svg+xml;base64,AAAA")).toBeNull();
    expect(keep("path", "href", "#dot")).toBeNull();
    expect(keep("path", "fill", "url(https://example.com/p.svg#g)")).toBeNull();
    expect(keep("path", "fill", "url(#a) url(//example.com)")).toBeNull();
    expect(keep("path", "mask", "url(data:image/png;base64,AAAA)")).toBeNull();
    expect(keep("g", "id", "a b")).toBeNull();
    expect(keep("g", "id", '"><script>')).toBeNull();
  });

  it("rejects values with markup, schemes, or escapes", () => {
    expect(keep("path", "fill", "javascript:alert(1)")).toBeNull();
    expect(keep("path", "d", "M0 0<script>")).toBeNull();
    expect(keep("text", "font-family", "\\75rl(x)")).toBeNull();
  });

  it("keeps the role colors and the opacities a dimmed block uses", () => {
    for (const role of ["attn", "mlp", "norm", "embed", "linear", "softmax", "cache", "panel"]) {
      expect(keep("rect", "style", `fill: var(--figure-${role})`)).toBe(`fill: var(--figure-${role})`);
    }
    expect(keep("rect", "fill-opacity", "0.4")).toBe("0.4");
    expect(keep("rect", "stroke-opacity", "0.4")).toBe("0.4");
    expect(sanitizeStyle("fill: var(--figure-attn); fill-opacity: 0.4; stroke-opacity: 0.4")).toBe(
      "fill: var(--figure-attn); fill-opacity: 0.4; stroke-opacity: 0.4",
    );
    // Step groups are found by id, and a step's note is its desc.
    expect(keep("g", "id", "step-2")).toBe("f1-step-2");
    expect(SVG_ELEMENTS.has("desc")).toBe(true);
  });

  it("reduces style to paint and text properties with plain values", () => {
    expect(sanitizeStyle("stroke: var(--figure-accent); fill:none")).toBe(
      "stroke: var(--figure-accent); fill: none",
    );
    expect(sanitizeStyle("font-family: var(--figure-serif); font-style: italic;")).toBe(
      "font-family: var(--figure-serif); font-style: italic",
    );
    expect(sanitizeStyle("position: fixed; inset: 0; fill: currentColor")).toBe("fill: currentColor");
    expect(sanitizeStyle("fill: url(https://example.com/x.svg#p)")).toBe("");
    expect(sanitizeStyle("background-image: url(x)")).toBe("");
    expect(sanitizeStyle("fill: expression(alert(1))")).toBe("");
    expect(sanitizeStyle("stroke: red; behavior: url(x.htc)")).toBe("stroke: red");
    expect(sanitizeStyle("fill: red !important; --x: 1")).toBe("");
    expect(keep("g", "style", "display: none")).toBeNull();
    expect(keep("g", "style", "opacity: 0.5")).toBe("opacity: 0.5");
  });
});
