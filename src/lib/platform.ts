import { isTauri } from "@/lib/api";

export const IS_MAC =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

/**
 * True when the window uses an overlay title bar, so the app bar must leave
 * room for the traffic lights and act as the drag region. tauri.conf.json sets
 * `titleBarStyle: "Overlay"`, which only macOS honors, so this is exactly
 * "inside Tauri on a Mac": no build flag to keep in sync.
 */
export const TITLEBAR_OVERLAY = isTauri() && IS_MAC;

/** Key glyphs for shortcut hints. */
export const MOD_KEY = IS_MAC ? "⌘" : "Ctrl";
export const SHIFT_KEY = IS_MAC ? "⇧" : "Shift";
export const ENTER_KEY = "↵";

export function isModKey(event: KeyboardEvent): boolean {
  return IS_MAC ? event.metaKey : event.ctrlKey;
}

/**
 * Removes browser behavior that does not belong in a desktop app: the web
 * context menu on chrome (kept on text you can select or edit) and
 * drag-and-drop navigation.
 */
export function installNativeBehavior(): () => void {
  const onContextMenu = (event: MouseEvent) => {
    const target = event.target as HTMLElement | null;
    const keepsMenu = target?.closest("input, textarea, [contenteditable='true'], .selectable, .cm-editor");
    if (!keepsMenu) event.preventDefault();
  };
  const onDrop = (event: DragEvent) => event.preventDefault();
  if (isTauri()) window.addEventListener("contextmenu", onContextMenu);
  window.addEventListener("dragover", onDrop);
  window.addEventListener("drop", onDrop);
  return () => {
    window.removeEventListener("contextmenu", onContextMenu);
    window.removeEventListener("dragover", onDrop);
    window.removeEventListener("drop", onDrop);
  };
}

/** Opens a web link in the system browser (the app window never navigates away). */
export function openExternal(url: string): void {
  if (!/^https?:\/\//.test(url)) return;
  if (isTauri()) {
    void import("@tauri-apps/plugin-opener").then(({ openUrl }) => openUrl(url)).catch(() => {});
  } else {
    window.open(url, "_blank", "noopener,noreferrer");
  }
}
