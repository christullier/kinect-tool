# Kinect Stream

A small local web viewer for an Xbox 360 Kinect. It streams RGB and depth frames
from `libfreenect` through a built-in Python HTTP server.

## Requirements

- macOS
- Xbox 360 Kinect
- `libfreenect`
- Python 3.10 or newer

Install the native library with Homebrew:

```sh
brew install libfreenect
```

Install Python dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Plug in the Kinect, then start the server:

```sh
python server.py
```

Open:

```text
http://localhost:8000
```

The page has Start and Stop controls plus three views: split, RGB only, and
depth only.

## Configuration

The server listens on `0.0.0.0:8000` by default. Override the bind address or
port with environment variables:

```sh
KINECT_HOST=127.0.0.1 KINECT_PORT=9000 python server.py
```

`PORT` is also honored for hosts that set it automatically.

## Endpoints

- `GET /` - web UI
- `GET /api/status` - stream status
- `POST /api/start` - start capture
- `POST /api/stop` - stop capture
- `GET /stream/rgb.mjpg` - RGB MJPEG stream
- `GET /stream/depth.mjpg` - depth MJPEG stream
- `GET /api/latest/rgb.png` - latest RGB frame
- `GET /api/latest/depth.png` - latest depth frame

## Notes

This is a local hardware tool. It does not include authentication, so do not
expose it directly to an untrusted network.
