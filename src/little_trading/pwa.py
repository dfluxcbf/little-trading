"""What makes the web app installable as a phone app (a PWA): the manifest, the service worker and the icons.

The icons are drawn here (a rising line on the accent colour) and encoded as PNG with the standard library, so
the package stays dependency-free and has no binary files. Android needs a 192 and a 512 pixel icon; the
maskable one fills the whole square and keeps the line inside the middle 80% that every launcher shape shows.
"""
from __future__ import annotations

import json
import math
import struct
import zlib
from functools import lru_cache

NAME = "little-trading dry run"
SHORT_NAME = "Trading"
THEME = "#2563eb"
BACKGROUND = "#0f1115"
ICONS = {  # path -> (size, maskable)
    "/icon-192.png": (192, False),
    "/icon-512.png": (512, False),
    "/icon-maskable-512.png": (512, True),
    "/apple-touch-icon.png": (180, True),
}

# The line, in a unit square (x right, y down): a dip, then a rise.
LINE = [(0.0, 0.70), (0.30, 0.48), (0.50, 0.62), (1.0, 0.18)]


def manifest() -> dict:
    return {
        "name": NAME, "short_name": SHORT_NAME, "id": "/", "start_url": "/", "scope": "/",
        "display": "standalone", "orientation": "any", "theme_color": THEME, "background_color": BACKGROUND,
        "description": "Fake money on real prices and news: what the dry-run trader holds, decides and learns.",
        "icons": [{"src": path, "sizes": f"{size}x{size}", "type": "image/png",
                   "purpose": "maskable" if maskable else "any"}
                  for path, (size, maskable) in ICONS.items() if path != "/apple-touch-icon.png"],
    }


def manifest_json() -> bytes:
    return json.dumps(manifest()).encode()


# Network first: the app shows live data, so the cache only lets the page open (with a "can't reach the server"
# state) when the phone is offline or the server is down. The API is never cached.
SERVICE_WORKER = """const CACHE = "little-trading-v1";
const SHELL = ["/", "/manifest.webmanifest", "/icon-192.png"];
self.addEventListener("install", e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL))); self.skipWaiting(); });
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || url.pathname.startsWith("/api/")) return;
  e.respondWith(fetch(e.request).then(r => {
    if (r.ok) { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); }
    return r;
  }).catch(() => caches.match(e.request).then(r => r || caches.match("/"))));
});
"""


def _hex(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _segment_distance(px: float, py: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _png(size: int, pixels: bytearray) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    stride = size * 4
    raw = b"".join(b"\x00" + bytes(pixels[y * stride:(y + 1) * stride]) for y in range(size))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


@lru_cache(maxsize=None)
def icon(size: int, maskable: bool) -> bytes:
    """The app icon as PNG: a white rising line on the accent colour, anti-aliased."""
    bg = _hex(THEME)
    # Where the line sits: the middle 60% (maskable: inside the 80% safe zone), a little lower than centre.
    margin = 0.24 if maskable else 0.20
    span = 1 - 2 * margin
    points = [((margin + x * span) * size, (margin + y * span) * size + 0.02 * size) for x, y in LINE]
    half = 0.045 * size  # half the line's width
    radius = 0.0 if maskable else 0.22 * size  # rounded corners (a maskable icon is cut by the launcher)
    pixels = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            cx, cy = x + 0.5, y + 0.5
            alpha = 1.0
            if radius:  # coverage of the rounded square
                qx = max(radius - cx, cx - (size - radius), 0.0)
                qy = max(radius - cy, cy - (size - radius), 0.0)
                alpha = max(0.0, min(1.0, radius - math.hypot(qx, qy) + 0.5))
            if alpha == 0.0:
                continue
            d = min(_segment_distance(cx, cy, a, b) for a, b in zip(points, points[1:]))
            line = max(0.0, min(1.0, half - d + 0.5))
            i = (y * size + x) * 4
            pixels[i:i + 4] = bytes((round(c + (255 - c) * line) for c in bg)) + bytes((round(255 * alpha),))
    return _png(size, pixels)


def asset(path: str) -> tuple[bytes, str] | None:
    """The body and content type of a PWA file, or None when `path` is not one."""
    if path == "/manifest.webmanifest":
        return manifest_json(), "application/manifest+json"
    if path == "/sw.js":
        return SERVICE_WORKER.encode(), "text/javascript; charset=utf-8"
    if path in ICONS:
        return icon(*ICONS[path]), "image/png"
    if path == "/favicon.ico":
        return icon(192, False), "image/png"
    return None
