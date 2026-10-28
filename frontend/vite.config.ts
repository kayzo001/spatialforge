import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend proxies /api so frontend and backend can run as separate dev
// servers; in production the backend serves frontend/dist directly.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    sourcemap: false,
    chunkSizeWarningLimit: 1600,
  },
});
