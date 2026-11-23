/** The bundled skill graphics by file name, resolved by Vite. */
const ART_FILES = import.meta.glob<string>("../assets/skills/*.webp", {
  eager: true,
  import: "default",
});

/** The image for a skill's `art` slug, or null when the slug is missing or names no bundled graphic. */
export function skillArtUrl(slug: string | undefined): string | null {
  if (!slug) return null;
  return ART_FILES[`../assets/skills/${slug}.webp`] ?? null;
}
