import { defineConfig } from "vite";

/** Slug = folder name under apps/ → Vite base `/<slug>/` for the combined Worker. */
const slug = "backprop";

export default defineConfig({
  base: `/${slug}/`,
  root: ".",
  publicDir: "public",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    host: "127.0.0.1",
    port: 43141,
    strictPort: true,
  },
  preview: {
    host: "127.0.0.1",
    port: 43141,
    strictPort: true,
  },
});
