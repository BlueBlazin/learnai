/** The bundled scene illustrations by file name, resolved by Vite. */
const SCENE_FILES = import.meta.glob<string>("../assets/scenes/*.webp", {
  eager: true,
  import: "default",
});

/** The image for a module's `scene` slug, or null when the slug is missing or names no bundled scene. */
export function sceneUrl(slug: string | undefined): string | null {
  if (!slug) return null;
  return SCENE_FILES[`../assets/scenes/${slug}.webp`] ?? null;
}
