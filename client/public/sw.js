// Минимальный service worker: кэширует статическую оболочку приложения,
// чтобы PWA открывалась даже при кратковременной потере Wi-Fi до домашнего
// сервера. Запросы к /api/* всегда идут в сеть — сервер обработки пазла
// работает только в реальном времени.
const CACHE_NAME = "puzzlevision-shell-v1";
const SHELL_FILES = ["/", "/manifest.webmanifest", "/icons/icon-192.png", "/icons/icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.pathname.startsWith("/api/")) return; // всегда сеть

  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).catch(() => cached);
    })
  );
});
