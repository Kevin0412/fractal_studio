from pathlib import Path
import time,urllib.request
from playwright.sync_api import sync_playwright
from PIL import Image
out=Path("render-output");out.mkdir(exist_ok=True)
preview=Path("docs/generated");preview.mkdir(parents=True,exist_ok=True)
for url in ["http://127.0.0.1:18080/api/system/check","http://127.0.0.1:5174"]:
    for _ in range(90):
        try:urllib.request.urlopen(url,timeout=2);break
        except Exception:time.sleep(1)
    else:raise RuntimeError("Service failed: "+url)
with sync_playwright() as p:
    browser=p.chromium.launch(args=["--enable-unsafe-swiftshader"])
    for name,size in [("desktop",(1440,900)),("mobile",(412,915))]:
        page=browser.new_page(viewport={"width":size[0],"height":size[1]},device_scale_factor=1)
        page.goto("http://127.0.0.1:5174",wait_until="domcontentloaded")
        page.wait_for_timeout(7000)
        page.screenshot(path=str(out/(name+".png")),full_page=True)
        page.close()
    browser.close()
im=Image.open(out/"desktop.png");im.thumbnail((1280,900));im.save(preview/"preview.webp",quality=88)
