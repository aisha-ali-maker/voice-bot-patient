const CACHE_NAME = 'voice-bot-cache-v1';
const APP_SHELL = [
  '/',
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
  // لا تضيفي ملفات كبيرة جداً هنا
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(APP_SHELL);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(self.clients.claim());
});

/* Strategy:
   - /upload (POST) => لا نتعامل معه هنا (غير GET)
   - /responses/* (audio files) => network-first ثم cache copy (حتى يمكن إعادة تشغيل الصوت offline بعد تحميله)
   - static assets => cache-first
   - navigation => network-first fallback to cache offline page if exists
*/

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  // Serve cached app shell
  if (APP_SHELL.includes(url.pathname) || url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        return cached || fetch(event.request);
      })
    );
    return;
  }

  // Audio responses: network first, then cache fallback
  if (url.pathname.startsWith('/responses/')) {
    event.respondWith(
      fetch(event.request)
        .then(resp => {
          // clone & store
          const copy = resp.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
          return resp;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Navigation (HTML pages) -> try network then fallback to cache
  if (event.request.mode === 'navigate' || url.pathname === '/') {
    event.respondWith(
      fetch(event.request).catch(() => caches.match('/'))
    );
    return;
  }

  // Default fallback: cache first then network
  event.respondWith(
    caches.match(event.request).then(cached => cached || fetch(event.request))
  );
});
