/** The bundled scene illustrations by file name, resolved by Vite. */
const SCENE_FILES = import.meta.glob<string>("../assets/scenes/*.webp", {
  eager: true,
  import: "default",
});

/**
 * Where a wide crop of a scene should hold on to, as a CSS `object-position`.
 * Scenes are composed around their center; one whose subject stands high in the
 * picture names its own point here so a banner does not cut it off.
 */
const SCENE_FOCUS: Record<string, string> = {
  trailblazer: "center 12%",
};

/** The focus point of a scene, or undefined for the default center. */
export function sceneFocus(slug: string | undefined): string | undefined {
  return slug ? SCENE_FOCUS[slug] : undefined;
}

/** The image for a module's `scene` slug, or null when the slug is missing or names no bundled scene. */
export function sceneUrl(slug: string | undefined): string | null {
  if (!slug) return null;
  return SCENE_FILES[`../assets/scenes/${slug}.webp`] ?? null;
}
