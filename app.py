"""3D Pano Viewer – view and export DJI 360 sphere panoramas.

Folder layout (next to the .exe):
  Panoramas/<any name>/   put each set of drone shots (or a .zip of them) here
  Exports/<name>/         3D models written by the Export button
  Cache/                  stitched panoramas (safe to delete)
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import cv2

import exporter
import stitcher

if getattr(sys, "frozen", False):
    ROOT = os.path.dirname(sys.executable)
    RES = sys._MEIPASS
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    RES = ROOT
PANOS = os.path.join(ROOT, "Panoramas")
EXPORTS = os.path.join(ROOT, "Exports")
CACHE = os.path.join(ROOT, "Cache")
for d in (PANOS, EXPORTS, CACHE):
    os.makedirs(d, exist_ok=True)
if sys.stderr is None:  # windowed exe has no console: log errors to a file
    sys.stdout = sys.stderr = open(os.path.join(CACHE, "log.txt"), "a", buffering=1)

jobs = {}  # name -> {"msg", "frac", "state": running|done|error}
jobs_lock = threading.Lock()
last_ping = [time.time()]


def safe_name(name):
    name = os.path.basename(name)
    if not name or name.startswith(".") or not os.path.isdir(os.path.join(PANOS, name)):
        raise ValueError("Unknown panorama")
    return name


def cache_path(name, width):
    return os.path.join(CACHE, f"{name}_{width}.jpg")


def thumb_path(name):
    return os.path.join(CACHE, f"{name}_thumb.jpg")


def best_cached(name):
    for w in (12288, 8192, 4096):
        p = cache_path(name, w)
        if os.path.exists(p):
            return p, w
    return None, None


def unpack_zips():
    for f in os.listdir(PANOS):
        p = os.path.join(PANOS, f)
        if f.lower().endswith(".zip") and os.path.isfile(p):
            target = os.path.join(PANOS, os.path.splitext(f)[0])
            if os.path.isdir(target):
                continue
            with zipfile.ZipFile(p) as z:
                names = [n for n in z.namelist() if n.lower().endswith(stitcher.IMAGE_EXTS)]
                os.makedirs(target, exist_ok=True)
                for n in names:
                    with z.open(n) as src, open(os.path.join(target, os.path.basename(n)), "wb") as dst:
                        shutil.copyfileobj(src, dst)


def list_panos():
    unpack_zips()
    out = []
    for f in sorted(os.listdir(PANOS)):
        p = os.path.join(PANOS, f)
        if not os.path.isdir(p) or f.startswith("."):
            continue
        imgs = stitcher.list_images(p)
        if not imgs:
            continue
        cached, w = best_cached(f)
        with jobs_lock:
            job = dict(jobs.get(f, {}))
        out.append({"name": f, "images": len(imgs), "stitched": bool(cached), "width": w,
                    "thumb": os.path.exists(thumb_path(f)), "job": job or None,
                    "exported": os.path.isdir(os.path.join(EXPORTS, f))})
    return out


def run_stitch(name, width):
    def cb(msg, frac):
        with jobs_lock:
            jobs[name].update(msg=msg, frac=frac if frac is not None else jobs[name]["frac"])

    try:
        pano = stitcher.stitch_folder(os.path.join(PANOS, name), width, cb)
        if pano.shape[1] != width:
            pano = cv2.resize(pano, (width, width // 2), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, 93])
        buf.tofile(cache_path(name, width))
        ok, tb = cv2.imencode(".jpg", cv2.resize(pano, (640, 320), interpolation=cv2.INTER_AREA))
        tb.tofile(thumb_path(name))
        with jobs_lock:
            jobs[name].update(state="done", msg="Done", frac=1.0)
    except Exception as e:
        traceback.print_exc()
        with jobs_lock:
            jobs[name].update(state="error", msg=str(e))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path, ctype):
        if not path or not os.path.exists(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def do_GET(self):
        self.route()

    def do_POST(self):
        self.route()

    def route(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self.send_file(os.path.join(RES, "web", "index.html"), "text/html; charset=utf-8")
            if u.path == "/favicon.ico":
                return self.send_file(os.path.join(RES, "icon.ico"), "image/x-icon")
            if u.path == "/api/ping":
                last_ping[0] = time.time()
                return self.send_json({"ok": True})
            if u.path == "/api/list":
                return self.send_json({"panos": list_panos(), "root": ROOT})
            if u.path == "/api/stitch":
                name = safe_name(q["name"])
                width = int(q.get("width", 8192))
                width = width if width in (4096, 8192, 12288) else 8192
                with jobs_lock:
                    if jobs.get(name, {}).get("state") == "running":
                        return self.send_json({"ok": True})
                    jobs[name] = {"state": "running", "msg": "Starting…", "frac": 0.0}
                threading.Thread(target=run_stitch, args=(name, width), daemon=True).start()
                return self.send_json({"ok": True})
            if u.path == "/api/status":
                name = safe_name(q["name"])
                with jobs_lock:
                    return self.send_json(jobs.get(name) or {"state": "idle"})
            if u.path.startswith("/pano/"):
                name = safe_name(u.path[6:])
                return self.send_file(best_cached(name)[0], "image/jpeg")
            if u.path.startswith("/thumb/"):
                name = safe_name(u.path[7:])
                return self.send_file(thumb_path(name), "image/jpeg")
            if u.path == "/api/export":
                name = safe_name(q["name"])
                src, _ = best_cached(name)
                if not src:
                    return self.send_json({"error": "Stitch the panorama first"}, 400)
                formats = q.get("formats", "glb,obj,equirect").split(",")
                out = os.path.join(EXPORTS, name)
                files = exporter.export_all(out, name, src, formats)
                return self.send_json({"ok": True, "folder": out, "files": [os.path.basename(f) for f in files]})
            if u.path == "/api/open":
                which = q.get("what", "panos")
                if which == "export":
                    path = os.path.join(EXPORTS, safe_name(q["name"]))
                elif which == "exports":
                    path = EXPORTS
                else:
                    path = PANOS
                os.startfile(path)
                return self.send_json({"ok": True})
            self.send_error(404)
        except (KeyError, ValueError) as e:
            self.send_json({"error": str(e)}, 400)
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": str(e)}, 500)


def find_edge():
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.environ.get("LOCALAPPDATA")):
        if base:
            p = os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe")
            if os.path.exists(p):
                return p
    return None


def main():
    server = ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("PANO_PORT", 0))), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    edge = find_edge()
    if os.environ.get("PANO_NOWINDOW"):
        last_ping[0] = time.time() + 10**9
    elif edge:
        profile = os.path.join(CACHE, ".window")
        subprocess.Popen([edge, f"--app={url}", f"--user-data-dir={profile}",
                          "--window-size=1400,880", "--no-first-run",
                          "--disable-features=Translate", "--no-default-browser-check"])
    else:
        webbrowser.open(url)
    # quit shortly after the window is closed (the page pings every 2s),
    # but never while a stitch is still running
    last_ping[0] = max(last_ping[0], time.time() + 30)
    while True:
        time.sleep(2)
        with jobs_lock:
            busy = any(j.get("state") == "running" for j in jobs.values())
        if not busy and time.time() - last_ping[0] > 12:
            break
    server.shutdown()


if __name__ == "__main__":
    main()
