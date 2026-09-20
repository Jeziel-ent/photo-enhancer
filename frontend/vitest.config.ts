import { defineConfig } from "vitest/config";

/**
 * Deliberately minimal, separate from vite.config.ts: this project's
 * frontend test coverage is scoped to pure utility/math functions only
 * (see docs/ARCHITECTURE.md's "Testing" section for the reasoning), so no
 * jsdom/browser environment, plugin, or DOM testing library is needed —
 * plain Node is enough for every test in src/**\/*.test.ts.
 */
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
