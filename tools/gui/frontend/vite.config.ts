import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the UI from Vite and proxies the API to the gateway, so
// the frontend hot-reloads without rebuilding. `npm run build` emits `dist/`,
// which the gateway serves itself -- that is the deployed path, and it means
// the whole GUI is one process on one host.
const gateway = process.env.SO_SNAKE_GATEWAY ?? "http://127.0.0.1:8770";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: gateway, changeOrigin: true }
    }
  },
  build: { outDir: "dist", emptyOutDir: true }
});
