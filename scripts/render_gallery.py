from pathlib import Path
import time
import urllib.request

from playwright.sync_api import sync_playwright
from PIL import Image

out = Path("render-output")
out.mkdir(exist_ok=True)
preview = Path("docs/generated")
preview.mkdir(parents=True, exist_ok=True)

for url in ["http://127.0.0.1:18080/api/system/check", "http://127.0.0.1:5174"]:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("Service failed: " + url)

# A mounted canvas alone is not evidence of a completed render.
# Require actual, varied pixel data from the active visible map frame.
frame_ready = """() => {
    const canvas = document.querySelector('.map-wrap canvas.frame.active');
    if (!canvas || !canvas.width || !canvas.height) return false;
    const ctx = canvas.getContext('2d');
    if (!ctx) return false;
    const pixels = new Set();
    for (let y = 1; y < 10; y++) {
        for (let x = 1; x < 10; x++) {
            const d = ctx.getImageData(Math.floor(canvas.width*x/10),
                                       Math.floor(canvas.height*y/10), 1, 1).data;
            if (d[3]) pixels.add(d.slice(0,3).join(','));
        }
    }
    return pixels.size > 2;
}"""

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--enable-unsafe-swiftshader"])
    for name, size in [("desktop", (1440, 900)), ("mobile", (412, 915))]:
        page = browser.new_page(
            viewport={"width": size[0], "height": size[1]},
            device_scale_factor=1,
            is_mobile=name == "mobile",
            has_touch=name == "mobile",
        )
        page.on("pageerror", lambda exc: print("Browser error:", exc, flush=True))
        page.goto("http://127.0.0.1:5174", wait_until="domcontentloaded")
        try:
            page.wait_for_function(frame_ready, timeout=120_000)
            page.screenshot(path=str(out / (name + ".png")), full_page=True)
        except Exception:
            page.screenshot(path=str(out / (name + "-failed.png")), full_page=True)
            raise
        finally:
            page.close()
    browser.close()

with Image.open(out / "desktop.png") as image:
    image.thumbnail((1280, 900))
    image.save(preview / "gallery.webp", quality=85, method=6)
