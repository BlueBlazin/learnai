/**
 * What an author's SVG figure may contain. Figures are inlined into the page
 * (so `currentColor` and the figure palette follow the theme), which means the
 * SVG is part of the document and must be inert: shapes and text only. This
 * module is the policy, as pure functions; `sanitizeSvg` applies it to a parsed
 * document.
 *
 * Everything is an allowlist. An element that is not listed is removed with
 * its subtree; an attribute that is not listed, or whose value could load or
 * run anything, is removed.
 */

/** Shapes, text, grouping, and paint servers. No script, style, foreignObject, image, a, or animation. */
export const SVG_ELEMENTS: ReadonlySet<string> = new Set([
  "svg",
  "g",
  "defs",
  "title",
  "desc",
  "path",
  "line",
  "polyline",
  "polygon",
  "rect",
  "circle",
  "ellipse",
  "text",
  "tspan",
  "marker",
  "clipPath",
  "mask",
  "linearGradient",
  "radialGradient",
  "stop",
  "symbol",
  "use",
]);

const GEOMETRY = [
  "viewBox",
  "preserveAspectRatio",
  "width",
  "height",
  "x",
  "y",
  "x1",
  "y1",
  "x2",
  "y2",
  "cx",
  "cy",
  "r",
  "rx",
  "ry",
  "d",
  "points",
  "dx",
  "dy",
  "rotate",
  "textLength",
  "lengthAdjust",
  "transform",
  "pathLength",
  "markerWidth",
  "markerHeight",
  "markerUnits",
  "refX",
  "refY",
  "orient",
  "clipPathUnits",
  "maskUnits",
  "maskContentUnits",
  "gradientUnits",
  "gradientTransform",
  "spreadMethod",
  "fx",
  "fy",
  "offset",
];

/** Presentation attributes, which are also the properties a `style` attribute may set. */
const PRESENTATION = [
  "fill",
  "fill-opacity",
  "fill-rule",
  "stroke",
  "stroke-width",
  "stroke-opacity",
  "stroke-dasharray",
  "stroke-dashoffset",
  "stroke-linecap",
  "stroke-linejoin",
  "stroke-miterlimit",
  "color",
  "opacity",
  "stop-color",
  "stop-opacity",
  "clip-rule",
  "paint-order",
  "vector-effect",
  "shape-rendering",
  "font-family",
  "font-size",
  "font-style",
  "font-weight",
  "font-variant-numeric",
  "letter-spacing",
  "text-anchor",
  "dominant-baseline",
];

/** Presentation attributes whose value may point at a paint server, marker, clip, or mask in the same file. */
const REFERENCING = ["clip-path", "mask", "marker-start", "marker-mid", "marker-end"];

const ACCESSIBILITY = ["role", "aria-label", "aria-hidden"];

const ATTRIBUTES: ReadonlySet<string> = new Set([
  ...GEOMETRY,
  ...PRESENTATION,
  ...REFERENCING,
  ...ACCESSIBILITY,
]);
const STYLE_PROPERTIES: ReadonlySet<string> = new Set(PRESENTATION);

const ID = /^[A-Za-z_][\w.-]*$/;
const LOCAL_URL = /url\(\s*(['"]?)#([A-Za-z_][\w.-]*)\1\s*\)/g;
/** Characters a geometry, paint, or text-style value needs; no quotes, slashes, colons, or escapes. */
const PLAIN_VALUE = /^[\w\s#%.,()+-]*$/;

/** Elements that may carry `href`, and only to a fragment of the same file. */
const HREF_ELEMENTS: ReadonlySet<string> = new Set(["use", "linearGradient", "radialGradient"]);

/**
 * Rewrites `url(#id)` references to the prefixed id and rejects any other
 * `url(...)`: returns null when the value refers outside the file.
 */
function localizeUrls(value: string, prefix: string): string | null {
  const rewritten = value.replace(LOCAL_URL, (_match, _quote, id: string) => `url(#${prefix}${id})`);
  // Whatever `url(` is left was not a same-file reference.
  const leftover = rewritten.replace(/url\(#[\w.-]+\)/g, "");
  return /url\s*\(/i.test(leftover) ? null : rewritten;
}

/** A `style` attribute reduced to the paint and text properties, with plain values only. */
export function sanitizeStyle(style: string): string {
  const kept: string[] = [];
  for (const declaration of style.split(";")) {
    const colon = declaration.indexOf(":");
    if (colon === -1) continue;
    const property = declaration.slice(0, colon).trim().toLowerCase();
    const value = declaration.slice(colon + 1).trim();
    if (!STYLE_PROPERTIES.has(property) || value === "") continue;
    if (!PLAIN_VALUE.test(value) || /url|expression|image|attr\s*\(/i.test(value)) continue;
    kept.push(`${property}: ${value}`);
  }
  return kept.join("; ");
}

/**
 * Decides one attribute. Returns the value to keep (possibly rewritten) or null
 * to remove the attribute. `prefix` is put in front of every id and every
 * reference to one, so two figures in one document never share an id.
 *
 * `name` is the qualified name as written (`xlink:href`, `stroke-width`).
 */
export function sanitizeAttribute(
  element: string,
  name: string,
  value: string,
  prefix: string,
): string | null {
  if (name === "id") return ID.test(value) ? `${prefix}${value}` : null;
  if (name === "href" || name === "xlink:href") {
    const target = value.trim();
    return HREF_ELEMENTS.has(element) && target.startsWith("#") && ID.test(target.slice(1))
      ? `#${prefix}${target.slice(1)}`
      : null;
  }
  if (name === "style") {
    const style = sanitizeStyle(value);
    return style === "" ? null : style;
  }
  if (!ATTRIBUTES.has(name)) return null;
  if (name === "aria-label") return value;
  const localized = localizeUrls(value, prefix);
  if (localized === null) return null;
  return PLAIN_VALUE.test(localized) ? localized : null;
}
