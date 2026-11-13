import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: { alias: { "@": path.resolve(rootDir, "./src") } },
  define: { __WHETSTONE_FIXTURES__: "true" },
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
