import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE || "./",
  resolve: {
    alias: {
      grimace: path.resolve(__dirname, "../grimace-main/src/index.ts"),
      "ts-deepmerge": path.resolve(__dirname, "node_modules/ts-deepmerge"),
    },
  },
  optimizeDeps: {
    exclude: ["@huggingface/transformers"],
  },
  worker: {
    format: "es",
  },
});
