#!/usr/bin/env python3
"""Web UI and HTTP streaming server for an Xbox 360 Kinect."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import numpy as np
from PIL import Image

from capture import HEIGHT, WIDTH, KinectCapture, depth_to_preview


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


class KinectService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._starting = False
        self._error: str | None = None
        self._last_rgb: np.ndarray | None = None
        self._last_depth: np.ndarray | None = None
        self._last_frame_at: float | None = None
        self._frame_count = 0

    def start(self) -> dict:
        with self._lock:
            if self._running or self._starting:
                return self.status()
            self._stop_event.clear()
            self._error = None
            self._last_rgb = None
            self._last_depth = None
            self._last_frame_at = None
            self._starting = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        with self._condition:
            self._running = False
            self._starting = False
            self._thread = None
            self._condition.notify_all()
            return self.status()

    def status(self) -> dict:
        now = time.time()
        with self._lock:
            age = None if self._last_frame_at is None else max(0.0, now - self._last_frame_at)
            return {
                "running": self._running,
                "starting": self._starting,
                "hasFrame": self._last_rgb is not None and self._last_depth is not None,
                "frameCount": self._frame_count,
                "lastFrameAt": self._last_frame_at,
                "lastFrameAgeSeconds": age,
                "error": self._error,
                "resolution": {"width": WIDTH, "height": HEIGHT},
            }

    def latest(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            rgb = None if self._last_rgb is None else self._last_rgb.copy()
            depth = None if self._last_depth is None else self._last_depth.copy()
            return rgb, depth

    def wait_for_frame(self, last_count: int, timeout_s: float = 5.0) -> tuple[int, np.ndarray | None, np.ndarray | None]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while (
                self._frame_count <= last_count
                and self._error is None
                and (self._running or self._starting)
                and not self._stop_event.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            if (
                self._frame_count <= last_count
                or self._error is not None
                or (not self._running and not self._starting)
            ):
                return self._frame_count, None, None
            rgb = None if self._last_rgb is None else self._last_rgb.copy()
            depth = None if self._last_depth is None else self._last_depth.copy()
            return self._frame_count, rgb, depth

    def _capture_loop(self) -> None:
        try:
            with KinectCapture() as kinect:
                with self._condition:
                    self._running = True
                    self._starting = False
                    self._error = None
                    self._condition.notify_all()
                while not self._stop_event.is_set():
                    rgb, depth = kinect.wait_for_frames(timeout_s=1.0)
                    with self._condition:
                        self._last_rgb = rgb
                        self._last_depth = depth
                        self._last_frame_at = time.time()
                        self._frame_count += 1
                        self._condition.notify_all()
                    kinect._got_rgb = False
                    kinect._got_depth = False
        except Exception as exc:
            with self._condition:
                self._error = str(exc)
                self._last_rgb = None
                self._last_depth = None
                self._last_frame_at = None
                self._running = False
                self._starting = False
                self._condition.notify_all()
        finally:
            with self._condition:
                self._running = False
                self._starting = False
                self._condition.notify_all()


SERVICE = KinectService()


def encode_rgb_jpeg(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=82)
    return buffer.getvalue()


def encode_depth_jpeg(depth: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    depth_to_preview(depth).convert("RGB").save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kinect Stream</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101113;
      color: #f4f5f6;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: #101113; }
    button, select { font: inherit; }
    .app { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid #2b2f34;
      background: #17191c;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 650; letter-spacing: 0; }
    .status { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; color: #b9c0c8; font-size: 13px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: #6d737c; display: inline-block; }
    .dot.running { background: #2fbf71; }
    .dot.starting { background: #d6a23a; }
    .dot.error { background: #e05d4f; }
    main { display: grid; grid-template-columns: 220px 1fr; min-height: 0; }
    aside {
      border-right: 1px solid #2b2f34;
      padding: 16px;
      background: #15171a;
      overflow: auto;
    }
    .controls { display: grid; gap: 10px; }
    .button-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    button, select {
      height: 38px;
      border: 1px solid #3a4149;
      background: #20252a;
      color: #f4f5f6;
      border-radius: 7px;
      padding: 0 11px;
    }
    button { cursor: pointer; }
    button:hover { background: #2a3037; }
    button.primary { background: #1c6b4a; border-color: #25845d; }
    button.danger { background: #6e2b2b; border-color: #91413c; }
    button:disabled { opacity: 0.55; cursor: default; }
    label { display: grid; gap: 6px; color: #b9c0c8; font-size: 12px; }
    .readout {
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid #2b2f34;
      display: grid;
      gap: 8px;
      color: #cbd1d7;
      font-size: 13px;
    }
    .error { color: #ffb4ac; overflow-wrap: anywhere; }
    .viewer { min-width: 0; min-height: 0; padding: 16px; display: grid; grid-template-rows: 1fr; }
    .streams { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; min-height: 0; }
    .streams.rgb-only { grid-template-columns: 1fr; }
    .streams.depth-only { grid-template-columns: 1fr; }
    .streams.rgb-only .depth-panel, .streams.depth-only .rgb-panel { display: none; }
    .panel {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      border: 1px solid #2b2f34;
      border-radius: 8px;
      overflow: hidden;
      background: #070808;
    }
    .panel-title {
      height: 38px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 12px;
      background: #17191c;
      border-bottom: 1px solid #2b2f34;
      color: #d8dde3;
      font-size: 13px;
    }
    .frame-wrap { display: grid; place-items: center; min-height: 0; padding: 10px; }
    img.stream {
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 100px);
      object-fit: contain;
      background: #030404;
    }
    @media (max-width: 820px) {
      header { align-items: flex-start; flex-direction: column; }
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid #2b2f34; }
      .streams { grid-template-columns: 1fr; }
      img.stream { max-height: 48vh; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>Kinect Stream</h1>
      <div class="status">
        <span id="statusDot" class="dot"></span>
        <span id="stateText">Idle</span>
        <span id="frameText">0 frames</span>
        <span id="ageText">No frame</span>
      </div>
    </header>
    <main>
      <aside>
        <div class="controls">
          <div class="button-row">
            <button id="startBtn" class="primary">Start</button>
            <button id="stopBtn" class="danger">Stop</button>
          </div>
          <label>
            View
            <select id="viewMode">
              <option value="split">Split</option>
              <option value="rgb-only">RGB only</option>
              <option value="depth-only">Depth only</option>
            </select>
          </label>
        </div>
        <div class="readout">
          <div>Resolution: <span id="resolutionText">640 x 480</span></div>
          <div>Server: <span id="serverText"></span></div>
          <div id="errorText" class="error"></div>
        </div>
      </aside>
      <section class="viewer">
        <div id="streams" class="streams">
          <section class="panel rgb-panel">
            <div class="panel-title"><span>RGB</span><span>MJPEG</span></div>
            <div class="frame-wrap"><img id="rgbStream" class="stream" alt="RGB stream"></div>
          </section>
          <section class="panel depth-panel">
            <div class="panel-title"><span>Depth</span><span>MJPEG</span></div>
            <div class="frame-wrap"><img id="depthStream" class="stream" alt="Depth stream"></div>
          </section>
        </div>
      </section>
    </main>
  </div>
  <script>
    const els = {
      statusDot: document.getElementById("statusDot"),
      stateText: document.getElementById("stateText"),
      frameText: document.getElementById("frameText"),
      ageText: document.getElementById("ageText"),
      startBtn: document.getElementById("startBtn"),
      stopBtn: document.getElementById("stopBtn"),
      viewMode: document.getElementById("viewMode"),
      streams: document.getElementById("streams"),
      rgbStream: document.getElementById("rgbStream"),
      depthStream: document.getElementById("depthStream"),
      resolutionText: document.getElementById("resolutionText"),
      serverText: document.getElementById("serverText"),
      errorText: document.getElementById("errorText")
    };

    let latestStatus = null;
    let requestBusy = false;

    els.serverText.textContent = window.location.host;
    els.rgbStream.src = "/stream/rgb.mjpg";
    els.depthStream.src = "/stream/depth.mjpg";

    async function api(path, options = {}) {
      const res = await fetch(path, options);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }

    function updateControls(status = latestStatus) {
      const running = Boolean(status && status.running);
      const starting = Boolean(status && status.starting);
      els.startBtn.disabled = requestBusy || running || starting;
      els.stopBtn.disabled = requestBusy || (!running && !starting);
    }

    function setBusy(isBusy) {
      requestBusy = isBusy;
      updateControls();
    }

    function renderStatus(status) {
      latestStatus = status;
      els.statusDot.className = "dot";
      if (status.error) els.statusDot.classList.add("error");
      else if (status.starting) els.statusDot.classList.add("starting");
      else if (status.running) els.statusDot.classList.add("running");
      els.stateText.textContent = status.error ? "Error" : status.starting ? "Starting" : status.running ? "Running" : "Stopped";
      els.frameText.textContent = `${status.frameCount} frames`;
      els.ageText.textContent = status.lastFrameAgeSeconds == null ? "No frame" : `${status.lastFrameAgeSeconds.toFixed(1)}s ago`;
      els.errorText.textContent = status.error || "";
      els.resolutionText.textContent = `${status.resolution.width} x ${status.resolution.height}`;
      updateControls(status);
    }

    async function refresh() {
      try {
        renderStatus(await api("/api/status"));
      } catch (err) {
        els.errorText.textContent = err.message;
      }
    }

    els.startBtn.addEventListener("click", async () => {
      setBusy(true);
      try { renderStatus(await api("/api/start", { method: "POST" })); }
      catch (err) { els.errorText.textContent = err.message; }
      finally { setBusy(false); }
    });

    els.stopBtn.addEventListener("click", async () => {
      setBusy(true);
      try { renderStatus(await api("/api/stop", { method: "POST" })); }
      catch (err) { els.errorText.textContent = err.message; }
      finally { setBusy(false); }
    });

    els.viewMode.addEventListener("change", () => {
      els.streams.className = `streams ${els.viewMode.value === "split" ? "" : els.viewMode.value}`;
    });

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class KinectRequestHandler(BaseHTTPRequestHandler):
    server_version = "KinectHTTP/1.0"

    def do_GET(self) -> None:
        path = self._path()
        if path == "/":
            self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send_json(SERVICE.status())
        elif path == "/api/latest/rgb.png":
            self._send_latest_rgb()
        elif path == "/api/latest/depth.png":
            self._send_latest_depth()
        elif path == "/stream/rgb.mjpg":
            self._send_stream("rgb")
        elif path == "/stream/depth.mjpg":
            self._send_stream("depth")
        else:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        path = self._path()
        try:
            if path == "/api/start":
                self._send_json(SERVICE.start())
            elif path == "/api/stop":
                self._send_json(SERVICE.stop())
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def _path(self) -> str:
        return unquote(self.path.split("?", 1)[0])

    def _send_latest_rgb(self) -> None:
        rgb, _depth = SERVICE.latest()
        if rgb is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "No RGB frame is available")
            return
        self._send_bytes(encode_png(Image.fromarray(rgb, mode="RGB")), "image/png")

    def _send_latest_depth(self) -> None:
        _rgb, depth = SERVICE.latest()
        if depth is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "No depth frame is available")
            return
        self._send_bytes(encode_png(depth_to_preview(depth)), "image/png")

    def _send_stream(self, kind: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        last_count = -1
        while True:
            try:
                last_count, rgb, depth = SERVICE.wait_for_frame(last_count)
                if kind == "rgb":
                    if rgb is None:
                        time.sleep(0.15)
                        continue
                    payload = encode_rgb_jpeg(rgb)
                else:
                    if depth is None:
                        time.sleep(0.15)
                        continue
                    payload = encode_depth_jpeg(depth)
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", status=status)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    host = os.environ.get("KINECT_HOST", DEFAULT_HOST)
    port = int(os.environ.get("KINECT_PORT", os.environ.get("PORT", str(DEFAULT_PORT))))
    server = ThreadingHTTPServer((host, port), KinectRequestHandler)
    print(f"Serving Kinect stream on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SERVICE.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
