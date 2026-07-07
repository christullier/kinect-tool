#!/usr/bin/env python3
"""Web UI and HTTP streaming server for the Xbox 360 Kinect capture script."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


def _reexec_with_py312_venv() -> None:
    if os.environ.get("KINECT_SKIP_PY312_REEXEC") == "1":
        return
    if sys.version_info < (3, 14):
        return

    project_dir = Path(__file__).resolve().parent
    venv_dir = project_dir / ".venv312"
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists() or not (venv_dir / ".kinect-ready").exists():
        return

    current = Path(sys.executable).resolve()
    target = venv_python.resolve()
    if current == target:
        return

    os.environ["KINECT_SKIP_PY312_REEXEC"] = "1"
    os.execv(str(target), [str(target), *sys.argv])


_reexec_with_py312_venv()

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kinect-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

try:
    import mediapipe as mp
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.vision import pose_landmarker
    from mediapipe.tasks.python.vision.core import vision_task_running_mode
except ImportError:
    mp = None
    base_options = None
    pose_landmarker = None
    vision_task_running_mode = None

from capture import (
    FREENECT_TILT_MAX,
    FREENECT_TILT_MIN,
    HEIGHT,
    WIDTH,
    KinectCapture,
    depth_to_preview,
    save_frames,
)


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
FRAMES_DIR = Path("frames")
POSE_MODEL_PATH = Path("models/pose_landmarker_lite.task")


class SkeletonTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pose = None
        self._error: str | None = None
        self._disabled_reason: str | None = None

        # MediaPipe 0.10.35 currently installs on Python 3.14, but its macOS
        # pose landmarker aborts the process while initializing the graph.
        if mp is not None and sys.version_info >= (3, 14):
            self._disabled_reason = (
                "MediaPipe is installed, but pose tracking is not usable under "
                "Python 3.14 here. Run the server with a Python 3.12 venv."
            )

    def status(self) -> dict:
        if mp is None or pose_landmarker is None:
            return {
                "available": False,
                "running": False,
                "error": "Install optional dependency: pip install mediapipe",
            }
        if self._disabled_reason:
            return {
                "available": False,
                "running": False,
                "error": self._disabled_reason,
            }
        if not POSE_MODEL_PATH.exists():
            return {
                "available": False,
                "running": False,
                "error": f"Missing pose model: {POSE_MODEL_PATH}",
            }
        with self._lock:
            return {
                "available": self._error is None,
                "running": self._pose is not None,
                "error": self._error,
            }

    def overlay(self, rgb: np.ndarray) -> np.ndarray:
        if self._disabled_reason:
            return rgb
        if mp is None or pose_landmarker is None or base_options is None or not POSE_MODEL_PATH.exists():
            return rgb
        with self._lock:
            try:
                if self._pose is None:
                    options = pose_landmarker.PoseLandmarkerOptions(
                        base_options=base_options.BaseOptions(
                            model_asset_path=str(POSE_MODEL_PATH),
                            delegate=base_options.BaseOptions.Delegate.CPU,
                        ),
                        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
                        num_poses=1,
                        min_pose_detection_confidence=0.5,
                        min_pose_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                        output_segmentation_masks=False,
                    )
                    self._pose = pose_landmarker.PoseLandmarker.create_from_options(options)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                results = self._pose.detect_for_video(image, int(time.time() * 1000))
                self._error = None
            except Exception as exc:
                self._error = str(exc)
                return rgb

        if not results.pose_landmarks:
            return rgb

        overlay = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(overlay)
        connections = pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS

        def point(landmarks: list, index: int) -> tuple[int, int] | None:
            landmark = landmarks[index]
            if getattr(landmark, "visibility", 1.0) < 0.45:
                return None
            x = int(np.clip(landmark.x, 0.0, 1.0) * (WIDTH - 1))
            y = int(np.clip(landmark.y, 0.0, 1.0) * (HEIGHT - 1))
            return x, y

        for landmarks in results.pose_landmarks:
            for connection in connections:
                a = point(landmarks, connection.start)
                b = point(landmarks, connection.end)
                if a and b:
                    draw.line((a, b), fill=(0, 255, 180), width=4)
            for index in range(len(landmarks)):
                p = point(landmarks, index)
                if p:
                    x, y = p
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 230, 60), outline=(0, 0, 0))

        return np.asarray(overlay)


SKELETON = SkeletonTracker()


class KinectService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._kinect: KinectCapture | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._starting = False
        self._error: str | None = None
        self._last_rgb: np.ndarray | None = None
        self._last_depth: np.ndarray | None = None
        self._last_frame_at: float | None = None
        self._frame_count = 0
        self._tilt_degrees = 0.0
        self._motor_available = False
        self._motor_error: str | None = "Stream is stopped"

    def start(self) -> dict:
        with self._lock:
            if self._running or self._starting:
                return self.status()
            self._stop_event.clear()
            self._error = None
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
        with self._lock:
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
                "tiltDegrees": self._tilt_degrees,
                "tiltRange": {"min": FREENECT_TILT_MIN, "max": FREENECT_TILT_MAX},
                "motor": {
                    "available": self._motor_available,
                    "error": self._motor_error,
                },
                "resolution": {"width": WIDTH, "height": HEIGHT},
                "skeleton": SKELETON.status(),
            }

    def latest(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            rgb = None if self._last_rgb is None else self._last_rgb.copy()
            depth = None if self._last_depth is None else self._last_depth.copy()
            return rgb, depth

    def wait_for_frame(self, last_count: int, timeout_s: float = 5.0) -> tuple[int, np.ndarray | None, np.ndarray | None]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._frame_count <= last_count and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            rgb = None if self._last_rgb is None else self._last_rgb.copy()
            depth = None if self._last_depth is None else self._last_depth.copy()
            return self._frame_count, rgb, depth

    def snapshot(self) -> dict:
        rgb, depth = self.latest()
        if rgb is None or depth is None:
            raise RuntimeError("No frame is available yet. Start the stream first.")
        stem = time.strftime("frame_%Y%m%d_%H%M%S")
        rgb_path, depth_raw_path, depth_preview_path = save_frames(rgb, depth, FRAMES_DIR, stem)
        return {
            "rgb": str(rgb_path),
            "depthRaw": str(depth_raw_path),
            "depthPreview": str(depth_preview_path),
        }

    def set_tilt(self, degrees: float | None = None, delta: float | None = None) -> dict:
        with self._lock:
            kinect = self._kinect
            if kinect is None or not self._running:
                raise RuntimeError("Start the stream before moving the Kinect motor.")
            target = self._tilt_degrees
            if degrees is not None:
                target = float(degrees)
            if delta is not None:
                target += float(delta)

        actual = kinect.set_tilt_degrees(target)
        with self._lock:
            self._tilt_degrees = actual
            return self.status()

    def recent_snapshots(self, limit: int = 40) -> list[dict]:
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        rgb_files = sorted(FRAMES_DIR.glob("*_rgb.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        snapshots = []
        for rgb_path in rgb_files[:limit]:
            stem = rgb_path.name.removesuffix("_rgb.png")
            depth_preview = FRAMES_DIR / f"{stem}_depth_preview.png"
            depth_raw = FRAMES_DIR / f"{stem}_depth_mm.npy"
            snapshots.append(
                {
                    "stem": stem,
                    "rgb": f"/frames/{rgb_path.name}",
                    "depthPreview": f"/frames/{depth_preview.name}" if depth_preview.exists() else None,
                    "depthRaw": f"/frames/{depth_raw.name}" if depth_raw.exists() else None,
                    "mtime": rgb_path.stat().st_mtime,
                }
            )
        return snapshots

    def _capture_loop(self) -> None:
        try:
            with KinectCapture() as kinect:
                with self._condition:
                    self._kinect = kinect
                    self._motor_available = kinect.motor_available
                    self._motor_error = kinect.motor_error
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
                self._running = False
                self._starting = False
                self._condition.notify_all()
        finally:
            with self._condition:
                self._kinect = None
                self._motor_available = False
                self._motor_error = "Stream is stopped"
                self._running = False
                self._starting = False
                self._condition.notify_all()


SERVICE = KinectService()


def encode_rgb_jpeg(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=82)
    return buffer.getvalue()


def encode_skeleton_jpeg(rgb: np.ndarray) -> bytes:
    return encode_rgb_jpeg(SKELETON.overlay(rgb))


def depth_to_dots(depth: np.ndarray, step: int = 3) -> Image.Image:
    valid = depth[depth > 0]
    if valid.size == 0:
        return Image.fromarray(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), mode="RGB")

    near = int(np.percentile(valid, 5))
    far = int(np.percentile(valid, 95))
    if far <= near:
        far = near + 1

    sampled = depth[::step, ::step]
    mask = sampled > 0
    normalized = np.clip((sampled.astype(np.float32) - near) / (far - near), 0.0, 1.0)
    brightness = ((1.0 - normalized) * 215 + 40).astype(np.uint8)

    dots = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    y_idx, x_idx = np.nonzero(mask)
    y_idx *= step
    x_idx *= step

    dot_values = brightness[mask]
    dots[y_idx, x_idx, 0] = dot_values
    dots[y_idx, x_idx, 1] = np.maximum(dot_values, 120)
    dots[y_idx, x_idx, 2] = 255
    return Image.fromarray(dots, mode="RGB")


def encode_depth_jpeg(depth: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    depth_to_dots(depth).save(buffer, format="JPEG", quality=88)
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
  <title>Kinect Control</title>
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
    main { display: grid; grid-template-columns: 260px 1fr; min-height: 0; }
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
    .checkbox-row {
      min-height: 38px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: #d8dde3;
      font-size: 13px;
    }
    input[type="checkbox"] { width: 16px; height: 16px; accent-color: #2fbf71; }
    .tilt-pad {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }
    .tilt-pad button {
      min-width: 0;
      padding: 0 8px;
    }
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
    .snapshots { margin-top: 18px; display: grid; gap: 8px; }
    .snapshots h2 { margin: 0; font-size: 13px; color: #d8dde3; }
    .snapshot {
      display: grid;
      gap: 5px;
      padding: 8px;
      border: 1px solid #2b2f34;
      border-radius: 7px;
      background: #1b1e22;
      font-size: 12px;
    }
    .snapshot span { color: #cbd1d7; overflow-wrap: anywhere; }
    .snapshot a { color: #8fc7ff; text-decoration: none; margin-right: 8px; }
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
      image-rendering: auto;
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
      <h1>Kinect Control</h1>
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
          <button id="snapshotBtn">Snapshot</button>
          <label>
            View
            <select id="viewMode">
              <option value="split">Split</option>
              <option value="rgb-only">RGB only</option>
              <option value="depth-only">Depth only</option>
            </select>
          </label>
          <label class="checkbox-row">
            <input id="skeletonToggle" type="checkbox">
            <span>Skeleton overlay</span>
          </label>
          <div class="tilt-pad">
            <button id="tiltUpBtn" title="Tilt up">Up</button>
            <button id="tiltCenterBtn" title="Center tilt">Center</button>
            <button id="tiltDownBtn" title="Tilt down">Down</button>
          </div>
        </div>
        <div class="readout">
          <div>Resolution: <span id="resolutionText">640 x 480</span></div>
          <div>Server: <span id="serverText"></span></div>
          <div>Tilt: <span id="tiltText">0 deg</span></div>
          <div>Motor: <span id="motorText">Stopped</span></div>
          <div>Skeleton: <span id="skeletonText">Checking</span></div>
          <div id="errorText" class="error"></div>
        </div>
        <div class="snapshots">
          <h2>Snapshots</h2>
          <div id="snapshotList"></div>
        </div>
      </aside>
      <section class="viewer">
        <div id="streams" class="streams">
          <section class="panel rgb-panel">
            <div class="panel-title"><span>RGB</span><span>MJPEG</span></div>
            <div class="frame-wrap"><img id="rgbStream" class="stream" alt="RGB stream"></div>
          </section>
          <section class="panel depth-panel">
            <div class="panel-title"><span>Depth Dots</span><span>MJPEG</span></div>
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
      snapshotBtn: document.getElementById("snapshotBtn"),
      tiltUpBtn: document.getElementById("tiltUpBtn"),
      tiltCenterBtn: document.getElementById("tiltCenterBtn"),
      tiltDownBtn: document.getElementById("tiltDownBtn"),
      viewMode: document.getElementById("viewMode"),
      skeletonToggle: document.getElementById("skeletonToggle"),
      streams: document.getElementById("streams"),
      rgbStream: document.getElementById("rgbStream"),
      depthStream: document.getElementById("depthStream"),
      resolutionText: document.getElementById("resolutionText"),
      serverText: document.getElementById("serverText"),
      tiltText: document.getElementById("tiltText"),
      motorText: document.getElementById("motorText"),
      skeletonText: document.getElementById("skeletonText"),
      errorText: document.getElementById("errorText"),
      snapshotList: document.getElementById("snapshotList")
    };

    let motorAvailable = false;

    els.serverText.textContent = window.location.host;
    function setRgbStream() {
      const path = els.skeletonToggle.checked ? "/stream/skeleton.mjpg" : "/stream/rgb.mjpg";
      if (!els.rgbStream.src.endsWith(path)) els.rgbStream.src = path;
    }
    setRgbStream();
    els.depthStream.src = "/stream/depth.mjpg";

    async function api(path, options = {}) {
      const res = await fetch(path, options);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }

    function setBusy(isBusy) {
      els.startBtn.disabled = isBusy;
      els.stopBtn.disabled = isBusy;
      els.snapshotBtn.disabled = isBusy;
      els.tiltUpBtn.disabled = isBusy || !motorAvailable;
      els.tiltCenterBtn.disabled = isBusy || !motorAvailable;
      els.tiltDownBtn.disabled = isBusy || !motorAvailable;
    }

    function renderStatus(status) {
      els.statusDot.className = "dot";
      if (status.error) els.statusDot.classList.add("error");
      else if (status.starting) els.statusDot.classList.add("starting");
      else if (status.running) els.statusDot.classList.add("running");
      els.stateText.textContent = status.error ? "Error" : status.starting ? "Starting" : status.running ? "Running" : "Stopped";
      els.frameText.textContent = `${status.frameCount} frames`;
      els.ageText.textContent = status.lastFrameAgeSeconds == null ? "No frame" : `${status.lastFrameAgeSeconds.toFixed(1)}s ago`;
      els.errorText.textContent = status.error || "";
      els.resolutionText.textContent = `${status.resolution.width} x ${status.resolution.height}`;
      els.tiltText.textContent = `${Number(status.tiltDegrees || 0).toFixed(1)} deg`;
      if (status.motor) renderMotorStatus(status.motor);
      if (status.skeleton) renderSkeletonStatus(status.skeleton);
    }

    function renderMotorStatus(motor) {
      motorAvailable = Boolean(motor.available);
      els.motorText.textContent = motorAvailable ? "Available" : (motor.error || "Unavailable");
      els.tiltUpBtn.disabled = !motorAvailable;
      els.tiltCenterBtn.disabled = !motorAvailable;
      els.tiltDownBtn.disabled = !motorAvailable;
    }

    function renderSkeletonStatus(skeleton) {
      els.skeletonToggle.disabled = !skeleton.available;
      els.skeletonText.textContent = skeleton.available ? (skeleton.running ? "Running" : "Available") : skeleton.error || "Unavailable";
      if (!skeleton.available && els.skeletonToggle.checked) {
        els.skeletonToggle.checked = false;
        setRgbStream();
      }
    }

    function renderSnapshots(items) {
      if (!items.length) {
        els.snapshotList.innerHTML = '<div class="snapshot"><span>No snapshots yet</span></div>';
        return;
      }
      els.snapshotList.innerHTML = items.map(item => `
        <div class="snapshot">
          <span>${item.stem}</span>
          <div>
            <a href="${item.rgb}" target="_blank">RGB</a>
            ${item.depthPreview ? `<a href="${item.depthPreview}" target="_blank">Depth</a>` : ""}
            ${item.depthRaw ? `<a href="${item.depthRaw}" target="_blank">Raw</a>` : ""}
          </div>
        </div>
      `).join("");
    }

    async function refresh() {
      try {
        renderStatus(await api("/api/status"));
      } catch (err) {
        els.errorText.textContent = err.message;
      }
    }

    async function refreshSnapshots() {
      try {
        renderSnapshots(await api("/api/snapshots"));
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

    els.snapshotBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        await api("/api/snapshot", { method: "POST" });
        await refreshSnapshots();
      } catch (err) {
        els.errorText.textContent = err.message;
      } finally {
        setBusy(false);
      }
    });

    async function moveTilt(payload) {
      setBusy(true);
      try {
        renderStatus(await api("/api/tilt", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        }));
      } catch (err) {
        els.errorText.textContent = err.message;
      } finally {
        setBusy(false);
      }
    }

    els.tiltUpBtn.addEventListener("click", () => moveTilt({ delta: 5 }));
    els.tiltCenterBtn.addEventListener("click", () => moveTilt({ degrees: 0 }));
    els.tiltDownBtn.addEventListener("click", () => moveTilt({ delta: -5 }));

    els.viewMode.addEventListener("change", () => {
      els.streams.className = `streams ${els.viewMode.value === "split" ? "" : els.viewMode.value}`;
    });

    els.skeletonToggle.addEventListener("change", setRgbStream);

    refresh();
    refreshSnapshots();
    setInterval(refresh, 1000);
    setInterval(refreshSnapshots, 5000);
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
        elif path == "/api/skeleton":
            self._send_json(SKELETON.status())
        elif path == "/api/snapshots":
            self._send_json(SERVICE.recent_snapshots())
        elif path == "/api/latest/rgb.png":
            self._send_latest_rgb()
        elif path == "/api/latest/depth.png":
            self._send_latest_depth()
        elif path == "/stream/rgb.mjpg":
            self._send_stream("rgb")
        elif path == "/stream/skeleton.mjpg":
            self._send_stream("skeleton")
        elif path == "/stream/depth.mjpg":
            self._send_stream("depth")
        elif path.startswith("/frames/"):
            self._send_frame_file(path)
        else:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        path = self._path()
        try:
            if path == "/api/start":
                self._send_json(SERVICE.start())
            elif path == "/api/stop":
                self._send_json(SERVICE.stop())
            elif path == "/api/snapshot":
                self._send_json(SERVICE.snapshot(), status=HTTPStatus.CREATED)
            elif path == "/api/tilt":
                payload = self._read_json()
                self._send_json(
                    SERVICE.set_tilt(
                        degrees=payload.get("degrees"),
                        delta=payload.get("delta"),
                    )
                )
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def _path(self) -> str:
        return unquote(self.path.split("?", 1)[0])

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8"))

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
                elif kind == "skeleton":
                    if rgb is None:
                        time.sleep(0.15)
                        continue
                    payload = encode_skeleton_jpeg(rgb)
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

    def _send_frame_file(self, path: str) -> None:
        name = Path(path.removeprefix("/frames/")).name
        file_path = FRAMES_DIR / name
        if not file_path.exists() or not file_path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "Frame file not found")
            return
        if file_path.suffix == ".png":
            content_type = "image/png"
        elif file_path.suffix == ".npy":
            content_type = "application/octet-stream"
        else:
            self._send_error_json(HTTPStatus.FORBIDDEN, "Unsupported frame file type")
            return
        self._send_bytes(file_path.read_bytes(), content_type)

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
    print(f"Serving Kinect UI on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server", flush=True)
    finally:
        SERVICE.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
