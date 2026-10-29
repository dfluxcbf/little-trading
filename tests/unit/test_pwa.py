import json
import struct
import threading
import urllib.request
import zlib

import pytest

from little_trading import pwa
from little_trading.web import Api, make_handler
from http.server import ThreadingHTTPServer

# --- REQ-000000020: Web app installable as a phone app ---


class NoServer:  # the PWA files need nothing from the server
    store = None


@pytest.fixture
def base():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Api(NoServer())))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def png_size(data: bytes) -> tuple[int, int]:
    assert data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR"
    zlib.decompress(data[data.index(b"IDAT") + 4:data.index(b"IEND") - 8])  # the pixels decode
    return struct.unpack(">II", data[16:24])


@pytest.mark.requirement("REQ-000000020")
def test_page_links_the_manifest_and_registers_the_service_worker(base):
    page = urllib.request.urlopen(base + "/").read().decode()
    assert '<link rel="manifest" href="/manifest.webmanifest">' in page
    assert 'serviceWorker.register("/sw.js")' in page and 'name="theme-color"' in page


@pytest.mark.requirement("REQ-000000020")
def test_manifest_has_what_android_needs_to_install_it(base):
    r = urllib.request.urlopen(base + "/manifest.webmanifest")
    assert r.headers["Content-Type"] == "application/manifest+json"
    m = json.loads(r.read())
    assert m["display"] == "standalone" and m["start_url"] == "/" and m["name"] and m["short_name"]
    sizes = {(i["sizes"], i["purpose"]) for i in m["icons"]}
    assert {("192x192", "any"), ("512x512", "any"), ("512x512", "maskable")} <= sizes
    for i in m["icons"]:
        got = urllib.request.urlopen(base + i["src"])
        assert got.headers["Content-Type"] == "image/png"
        n = int(i["sizes"].split("x")[0])
        assert png_size(got.read()) == (n, n)


@pytest.mark.requirement("REQ-000000020")
def test_service_worker_never_caches_the_api(base):
    sw = urllib.request.urlopen(base + "/sw.js")
    assert sw.headers["Content-Type"].startswith("text/javascript")
    body = sw.read().decode()
    assert 'url.pathname.startsWith("/api/")) return' in body and 'addEventListener("fetch"' in body


@pytest.mark.requirement("REQ-000000020")
def test_icons_are_opaque_where_the_launcher_cuts_a_maskable_one():
    data = pwa.icon(64, True)
    raw = zlib.decompress(data[data.index(b"IDAT") + 4:data.index(b"IEND") - 8])
    alphas = {raw[1 + y * (64 * 4 + 1) + x * 4 + 3] for y in range(64) for x in range(64)}
    assert alphas == {255}  # full bleed: no transparent corners
    corner = zlib.decompress((d := pwa.icon(64, False))[d.index(b"IDAT") + 4:d.index(b"IEND") - 8])[1 + 3]
    assert corner == 0  # the plain icon has rounded corners
