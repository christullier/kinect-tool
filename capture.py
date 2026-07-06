#!/usr/bin/env python3
"""Capture RGB and depth frames from an Xbox 360 Kinect via libfreenect."""

from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

FREENECT_DEVICE_MOTOR = 0x01
FREENECT_DEVICE_CAMERA = 0x02
FREENECT_RESOLUTION_MEDIUM = 1
FREENECT_VIDEO_RGB = 0
FREENECT_DEPTH_MM = 5
FREENECT_TILT_MIN = -30.0
FREENECT_TILT_MAX = 30.0

WIDTH, HEIGHT = 640, 480
RGB_BYTES = WIDTH * HEIGHT * 3
DEPTH_BYTES = WIDTH * HEIGHT * 2


def _load_libfreenect() -> ctypes.CDLL:
    for name in (
        "libfreenect.0.7.5.dylib",
        "libfreenect.0.dylib",
        "libfreenect.dylib",
        "freenect",
    ):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError(
        "libfreenect not found. Install with: brew install libfreenect"
    )


class FrameMode(ctypes.Structure):
    _fields_ = [
        ("resolution", ctypes.c_int),
        ("video_format", ctypes.c_int),
        ("depth_format", ctypes.c_int),
        ("bytes", ctypes.c_int),
        ("width", ctypes.c_int16),
        ("height", ctypes.c_int16),
        ("data_bits_per_pixel", ctypes.c_int8),
        ("padding_bits_per_pixel", ctypes.c_int8),
        ("framerate", ctypes.c_int8),
        ("is_valid", ctypes.c_int8),
    ]


VideoCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)
DepthCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)


class KinectCapture:
    def __init__(self) -> None:
        self._lib = _load_libfreenect()
        self._ctx = ctypes.c_void_p()
        self._dev = ctypes.c_void_p()
        self._device_lock = threading.RLock()
        self._motor_available = False
        self._motor_error: str | None = None
        self._rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self._depth = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
        self._got_rgb = False
        self._got_depth = False
        self._video_cb = VideoCallback(self._on_video)
        self._depth_cb = DepthCallback(self._on_depth)
        self._bind()

    def _bind(self) -> None:
        lib = self._lib
        lib.freenect_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        lib.freenect_init.restype = ctypes.c_int
        lib.freenect_shutdown.argtypes = [ctypes.c_void_p]
        lib.freenect_shutdown.restype = ctypes.c_int
        lib.freenect_select_subdevices.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.freenect_num_devices.argtypes = [ctypes.c_void_p]
        lib.freenect_num_devices.restype = ctypes.c_int
        lib.freenect_open_device.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        lib.freenect_open_device.restype = ctypes.c_int
        lib.freenect_close_device.argtypes = [ctypes.c_void_p]
        lib.freenect_close_device.restype = ctypes.c_int
        lib.freenect_find_video_mode.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.freenect_find_video_mode.restype = FrameMode
        lib.freenect_find_depth_mode.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.freenect_find_depth_mode.restype = FrameMode
        lib.freenect_set_video_mode.argtypes = [ctypes.c_void_p, FrameMode]
        lib.freenect_set_video_mode.restype = ctypes.c_int
        lib.freenect_set_depth_mode.argtypes = [ctypes.c_void_p, FrameMode]
        lib.freenect_set_depth_mode.restype = ctypes.c_int
        lib.freenect_set_video_callback.argtypes = [ctypes.c_void_p, VideoCallback]
        lib.freenect_set_depth_callback.argtypes = [ctypes.c_void_p, DepthCallback]
        lib.freenect_start_video.argtypes = [ctypes.c_void_p]
        lib.freenect_start_video.restype = ctypes.c_int
        lib.freenect_start_depth.argtypes = [ctypes.c_void_p]
        lib.freenect_start_depth.restype = ctypes.c_int
        lib.freenect_stop_video.argtypes = [ctypes.c_void_p]
        lib.freenect_stop_depth.argtypes = [ctypes.c_void_p]
        lib.freenect_process_events.argtypes = [ctypes.c_void_p]
        lib.freenect_process_events.restype = ctypes.c_int
        lib.freenect_set_tilt_degs.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.freenect_set_tilt_degs.restype = ctypes.c_int

    def _on_video(self, _dev, data, _timestamp) -> None:
        ctypes.memmove(self._rgb.ctypes.data, data, RGB_BYTES)
        self._got_rgb = True

    def _on_depth(self, _dev, data, _timestamp) -> None:
        ctypes.memmove(self._depth.ctypes.data, data, DEPTH_BYTES)
        self._got_depth = True

    def __enter__(self) -> KinectCapture:
        if self._lib.freenect_init(ctypes.byref(self._ctx), None) < 0:
            raise RuntimeError("freenect_init failed")
        # Try camera + tilt motor first. If the motor interface cannot be
        # claimed, fall back to camera-only so RGB/depth still work.
        if self._lib.freenect_num_devices(self._ctx) < 1:
            self._lib.freenect_shutdown(self._ctx)
            raise RuntimeError("No Kinect found. Is it plugged in and USB access approved?")
        self._open_device(FREENECT_DEVICE_CAMERA | FREENECT_DEVICE_MOTOR)
        video_mode = self._lib.freenect_find_video_mode(
            FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB
        )
        depth_mode = self._lib.freenect_find_depth_mode(
            FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_MM
        )
        if self._lib.freenect_set_depth_mode(self._dev, depth_mode) < 0:
            raise RuntimeError("Failed to set depth mode")
        if self._lib.freenect_set_video_mode(self._dev, video_mode) < 0:
            raise RuntimeError("Failed to set video mode")
        self._lib.freenect_set_depth_callback(self._dev, self._depth_cb)
        self._lib.freenect_set_video_callback(self._dev, self._video_cb)
        if self._lib.freenect_start_depth(self._dev) < 0:
            raise RuntimeError("Failed to start depth stream")
        if self._lib.freenect_start_video(self._dev) < 0:
            raise RuntimeError("Failed to start video stream")
        return self

    def _open_device(self, subdevices: int) -> None:
        self._lib.freenect_select_subdevices(self._ctx, subdevices)
        if self._lib.freenect_open_device(self._ctx, ctypes.byref(self._dev), 0) >= 0:
            self._motor_available = bool(subdevices & FREENECT_DEVICE_MOTOR)
            if self._motor_available:
                self._motor_error = None
            elif self._motor_error is None:
                self._motor_error = "Motor subdevice unavailable"
            return

        if subdevices & FREENECT_DEVICE_MOTOR:
            self._dev = ctypes.c_void_p()
            self._motor_available = False
            self._motor_error = "Failed to open Kinect motor subdevice; using camera-only mode"
            self._open_device(FREENECT_DEVICE_CAMERA)
            return

        self._lib.freenect_shutdown(self._ctx)
        raise RuntimeError("Failed to open Kinect")

    def __exit__(self, *_args) -> None:
        if self._dev.value:
            self._lib.freenect_stop_depth(self._dev)
            self._lib.freenect_stop_video(self._dev)
            self._lib.freenect_close_device(self._dev)
        if self._ctx.value:
            self._lib.freenect_shutdown(self._ctx)

    def wait_for_frames(self, timeout_s: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._device_lock:
                if self._lib.freenect_process_events(self._ctx) < 0:
                    break
            if self._got_rgb and self._got_depth:
                return self._rgb.copy(), self._depth.copy()
            time.sleep(0.01)
        missing = []
        if not self._got_rgb:
            missing.append("RGB")
        if not self._got_depth:
            missing.append("depth")
        raise TimeoutError(f"Timed out waiting for frames: {', '.join(missing)}")

    def set_tilt_degrees(self, degrees: float) -> float:
        angle = max(FREENECT_TILT_MIN, min(FREENECT_TILT_MAX, float(degrees)))
        if not self._dev.value:
            raise RuntimeError("Kinect is not open")
        if not self._motor_available:
            raise RuntimeError(self._motor_error or "Kinect motor is not available")
        with self._device_lock:
            if self._lib.freenect_set_tilt_degs(self._dev, angle) < 0:
                raise RuntimeError("Failed to move Kinect motor")
        return angle

    @property
    def motor_available(self) -> bool:
        return self._motor_available

    @property
    def motor_error(self) -> str | None:
        return self._motor_error

def depth_to_preview(depth_mm: np.ndarray) -> Image.Image:
    """Map millimeter depth to an 8-bit grayscale preview image."""
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return Image.fromarray(np.zeros((HEIGHT, WIDTH), dtype=np.uint8), mode="L")
    near = int(np.percentile(valid, 5))
    far = int(np.percentile(valid, 95))
    if far <= near:
        far = near + 1
    clipped = np.clip(depth_mm, near, far)
    preview = ((clipped - near) * 255 / (far - near)).astype(np.uint8)
    preview[depth_mm == 0] = 0
    return Image.fromarray(preview, mode="L")


def save_frames(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    out_dir: Path,
    stem: str,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = out_dir / f"{stem}_rgb.png"
    depth_raw_path = out_dir / f"{stem}_depth_mm.npy"
    depth_preview_path = out_dir / f"{stem}_depth_preview.png"

    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    np.save(depth_raw_path, depth_mm)
    depth_to_preview(depth_mm).save(depth_preview_path)
    return rgb_path, depth_raw_path, depth_preview_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("frames"),
        help="Output directory (default: frames/)",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="Number of frame pairs to capture (default: 1)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between captures when count > 1 (default: 0.5)",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Output filename prefix (default: frame)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with KinectCapture() as kinect:
            for i in range(args.count):
                rgb, depth = kinect.wait_for_frames()
                stem = f"{args.prefix}_{i:04d}" if args.count > 1 else args.prefix
                rgb_path, depth_raw_path, depth_preview_path = save_frames(
                    rgb, depth, args.output, stem
                )
                print(f"RGB:            {rgb_path}")
                print(f"Depth (raw):    {depth_raw_path}")
                print(f"Depth (preview): {depth_preview_path}")
                kinect._got_rgb = False
                kinect._got_depth = False
                if i + 1 < args.count:
                    time.sleep(args.interval)
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
