// ARIA Service Worker — offline support + background sync
const CACHE  = "aria-v7";
const SHELL  = ["/", "/index.html", "/manifest.json", "/icon-192.png"];

// Install — cache the app shell
self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

// Activate — clean up old caches
self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch — serve from cache when offline, network-first for API calls
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);

  // ── NEVER intercept API / auth / WebSocket requests ──────────────────────
  // These are all local server calls that must reach the backend directly.
  // Service Workers cannot proxy SSE streaming responses (both GET and POST) —
  // Chrome buffers the entire body before handing it to the page, which breaks
  // the notifications stream, chat stream, search stream, etc.
  // The SW's offline 503 fallback is more harmful than no fallback here.
  if (
    url.pathname.startsWith("/api/") ||
    url.pathname.startsWith("/auth/") ||
    url.pathname.startsWith("/ws/")
  ) return;

  // ── Also skip non-GET requests for all other paths ────────────────────────
  if (e.request.method !== "GET") return;

  // For app shell — cache-first with network fallback
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(response => {
        // Only cache HTML navigation — NEVER cache JS/CSS scripts.
        // JS files change on every Vite build; caching them serves stale code.
        if (response.ok && e.request.destination === "document") {
          const clone = response.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return response;
      }).catch(() => {
        // Offline fallback for navigation
        if (e.request.mode === "navigate") {
          return caches.match("/index.html");
        }
        return new Response("Offline", { status: 503 });
      });
    })
  );
});

// Background sync — queue failed API calls and retry when online
self.addEventListener("sync", e => {
  if (e.tag === "aria-sync") {
    e.waitUntil(flushQueue());
  }
});

const ACTION_QUEUE_KEY = "aria-action-queue";

async function flushQueue() {
  const queue = await getQueue();
  if (!queue.length) return;
  const remaining = [];
  for (const item of queue) {
    try {
      const response = await fetch(item.url, {
        method:  item.method,
        headers: item.headers,
        body:    item.body,
      });
      if (!response.ok) remaining.push(item);
    } catch {
      remaining.push(item);
    }
  }
  await setQueue(remaining);
}

async function getQueue() {
  const cache = await caches.open(CACHE);
  const match = await cache.match("/_queue");
  if (!match) return [];
  try { return await match.json(); } catch { return []; }
}

async function setQueue(queue) {
  const cache = await caches.open(CACHE);
  await cache.put("/_queue", new Response(JSON.stringify(queue)));
}

// Push notifications from server (Web Push API)
self.addEventListener("push", e => {
  if (!e.data) return;
  try {
    const data = e.data.json();
    e.waitUntil(
      self.registration.showNotification(`ARIA — ${data.title || "Alert"}`, {
        body:    data.body || "",
        icon:    "/icon-192.png",
        badge:   "/icon-192.png",
        tag:     data.id || "aria",
        data:    { action: data.action || "/" },
        vibrate: [200, 100, 200],
      })
    );
  } catch { /* malformed push data */ }
});

// Notification click — open ARIA and navigate to the right screen
self.addEventListener("notificationclick", e => {
  e.notification.close();
  const action = e.notification.data?.action || "/";
  e.waitUntil(
    clients.matchAll({ type: "window" }).then(clientList => {
      for (const client of clientList) {
        if (client.url.includes("localhost") && "focus" in client) {
          client.focus();
          client.navigate(action);
          return;
        }
      }
      clients.openWindow(action);
    })
  );
});
