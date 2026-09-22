import { defineConfig } from "vite";

// Сервер раздаёт собранный клиент статикой (server/app/main.py монтирует
// client/dist), поэтому в проде base — корень. В dev-режиме запросы к /api
// проксируются на локальный FastAPI (см. README про запуск в разработке).
export default defineConfig({
  base: "/",
  server: {
    host: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
