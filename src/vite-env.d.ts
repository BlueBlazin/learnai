/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "http" selects the devbridge transport in a plain browser. */
  readonly VITE_BACKEND?: string;
  /** Devbridge origin; defaults to http://127.0.0.1:1421. */
  readonly VITE_BACKEND_URL?: string;
}

/** False in builds made for the Tauri shell: the fixture backend and its data are left out. */
declare const __WHETSTONE_FIXTURES__: boolean;
