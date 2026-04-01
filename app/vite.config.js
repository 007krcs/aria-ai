import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 9177,
    strictPort: true,
    proxy: {
      "/api":  { target: "http://localhost:8000", changeOrigin: true, proxyTimeout: 120000, timeout: 120000 },
      "/auth": { target: "http://localhost:8000", changeOrigin: true, proxyTimeout: 120000, timeout: 120000 },
      "/ws":   { target: "ws://localhost:8000",   ws: true },
    },
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target:    ["es2021", "chrome105", "safari13"],
    outDir:    "dist",
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          "vendor-highlight": ["highlight.js"],
          "vendor-marked":    ["marked"],
        },
      },
    },
  },
});
