/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "http" selects the devbridge transport in a plain browser. */
  readonly VITE_BACKEND?: string;
  /** Devbridge origin; defaults to http://127.0.0.1:1421. */
  readonly VITE_BACKEND_URL?: string;
  /** "1" when tauri.conf.json uses an overlay title bar (reserves room for traffic lights). */
  readonly VITE_TITLEBAR_OVERLAY?: string;
}
