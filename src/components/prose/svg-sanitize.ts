import { SVG_ELEMENTS, sanitizeAttribute } from "./svg-policy";

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * The largest scale a figure is shown at. A drawing 480 units wide then exactly
 * fills the 640px measure, and in the narrowest statement pane (about 460px) it
 * is shown at just under 1:1. Text scales with the drawing up to 14 pixels and
 * stops there (`figure-text.ts`), so text drawn at 13 units stays between
 * roughly 12 and 14 pixels.
 */
const MAX_SCALE = 4 / 3;

/** Applies the policy to one element and, recursively, to what it contains. */
function clean(element: Element, prefix: string): void {
  for (const attribute of [...element.attributes]) {
    const value = sanitizeAttribute(element.localName, attribute.name, attribute.value, prefix);
    if (value === null) element.removeAttributeNode(attribute);
    else attribute.value = value;
  }
  for (const child of [...element.childNodes]) {
    if (child.nodeType === Node.TEXT_NODE) continue;
    if (
      child.nodeType === Node.ELEMENT_NODE &&
      (child as Element).namespaceURI === SVG_NS &&
      SVG_ELEMENTS.has((child as Element).localName)
    ) {
      clean(child as Element, prefix);
    } else {
      // Unknown elements, comments, processing instructions, CDATA: gone, subtree and all.
      child.remove();
    }
  }
}

/** The figure's natural size in CSS pixels: its viewBox, or failing that its width and height. */
function naturalSize(svg: Element): { width: number; height: number } | null {
  const box = (svg.getAttribute("viewBox") ?? "")
    .trim()
    .split(/[\s,]+/)
    .map(Number);
  const [, , boxWidth, boxHeight] = box;
  if (box.length === 4 && boxWidth && boxHeight && boxWidth > 0 && boxHeight > 0) {
    return { width: boxWidth, height: boxHeight };
  }
  const width = Number.parseFloat(svg.getAttribute("width") ?? "");
  const height = Number.parseFloat(svg.getAttribute("height") ?? "");
  return width > 0 && height > 0 ? { width, height } : null;
}

/**
 * Parses an author's SVG and returns an inert copy that is safe to put in the
 * page, or null when the source is not an SVG document. See `svg-policy.ts`
 * for what survives. The result is a DOM node, inserted as a node: the cleaned
 * tree is never serialized and parsed again.
 *
 * The root loses its `width` and `height` and is sized by CSS instead: as wide
 * as the measure allows, up to `MAX_SCALE` times its natural size.
 */
export function sanitizeSvg(source: string, prefix: string): SVGSVGElement | null {
  const parsed = new DOMParser().parseFromString(source, "image/svg+xml");
  const root = parsed.documentElement;
  if (root.namespaceURI !== SVG_NS || root.localName !== "svg" || parsed.querySelector("parsererror")) {
    return null;
  }
  const size = naturalSize(root);
  clean(root, prefix);
  const svg = document.importNode(root, true) as unknown as SVGSVGElement;
  svg.removeAttribute("width");
  svg.removeAttribute("height");
  if (size) {
    if (!svg.hasAttribute("viewBox")) svg.setAttribute("viewBox", `0 0 ${size.width} ${size.height}`);
    svg.style.maxWidth = `${size.width * MAX_SCALE}px`;
  }
  return svg;
}

/** The text of a `data:image/svg+xml;base64,` URL, or null for any other URL. */
export function svgFromDataUrl(url: string): string | null {
  const prefix = "data:image/svg+xml;base64,";
  if (!url.startsWith(prefix)) return null;
  const binary = atob(url.slice(prefix.length));
  return new TextDecoder().decode(Uint8Array.from(binary, (char) => char.charCodeAt(0)));
}
