#!/usr/bin/env python3
import argparse, io, os, time
from typing import Optional, Tuple

import cv2
import numpy as np
import requests

API_TIMEOUT = (3.0, 10.0)  # (connect, read)

def _open_cam(idx: int, w: int = 1280, h: int = 720, retries: int = 10) -> Optional[cv2.VideoCapture]:
    if idx is None or idx < 0:
        return None
    cap = None
    for _ in range(retries):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            # Try to set resolution; ignore if backend refuses
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            return cap
        if cap:
            cap.release()
        time.sleep(0.2)
    return None

def _read_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame

def _encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

def _hstack_or_single(grip: Optional[np.ndarray], ext: Optional[np.ndarray], comp_h: int) -> np.ndarray:
    if grip is None and ext is None:
        raise RuntimeError("No frames available")
    if ext is None:
        img = grip
    elif grip is None:
        img = ext
    else:
        # Pad heights then hstack
        h = max(grip.shape[0], ext.shape[0])
        def pad_to_h(x):
            if x.shape[0] == h:
                return x
            pad = h - x.shape[0]
            return cv2.copyMakeBorder(x, 0, pad, 0, 0, cv2.BORDER_REPLICATE)
        img = np.hstack([pad_to_h(grip), pad_to_h(ext)])

    # Resize to requested composite height (preserve aspect)
    ratio = comp_h / img.shape[0]
    comp_w = int(round(img.shape[1] * ratio))
    return cv2.resize(img, (comp_w, comp_h), interpolation=cv2.INTER_AREA)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="http://HOST:PORT")
    p.add_argument("--prompt", required=True)
    p.add_argument("--gripper-cam", type=int, default=0)
    p.add_argument("--external-cam", type=int, default=-1, help="-1 disables external cam")
    p.add_argument("--hz", type=float, default=5.0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--scale", type=float, default=12.0)
    p.add_argument("--comp-h", type=int, default=720)
    p.add_argument("--cam-w", type=int, default=1280)
    p.add_argument("--cam-h", type=int, default=720)
    p.add_argument("--print-every", type=int, default=10, help="print loop rate every N steps")
    args = p.parse_args()

    # Health
    try:
        r = requests.get(f"{args.server}/health", timeout=API_TIMEOUT)
        print("Health:", r.json())
    except Exception as e:
        print("⚠️  Health check failed:", e)

    print("📹 Initializing camera(s)...")
    grip_cap = _open_cam(args.gripper_cam, args.cam_w, args.cam_h)
    ext_cap = _open_cam(args.external_cam, args.cam_w, args.cam_h) if args.external_cam >= 0 else None

    if grip_cap is None and ext_cap is None:
        raise RuntimeError("No camera opened (gripper and external both unavailable). "
                           "Try changing indexes or unplug/plug cameras.")

    if ext_cap is None:
        print("✅ Running in SINGLE-CAM mode (gripper only).")
    else:
        print("✅ Running in DUAL-CAM mode (gripper + external).")

    period = 1.0 / max(0.1, args.hz)
    t_end = time.time() + args.duration

    step = 0
    last_print = time.perf_counter()
    since_print = 0

    while time.time() < t_end:
        t0 = time.perf_counter()

        grip = _read_frame(grip_cap) if grip_cap else None
        ext = _read_frame(ext_cap) if ext_cap else None

        if grip is None and ext is None:
            # brief backoff; keep loop alive
            time.sleep(0.01)
            continue

        comp = _hstack_or_single(grip, ext, args.comp_h)
        jpg = _encode_jpeg(comp, quality=88)

        files = {
            "image": ("frame.jpg", io.BytesIO(jpg), "image/jpeg")
        }
        data = {"prompt": args.prompt}

        t_req0 = time.perf_counter()
        try:
            resp = requests.post(f"{args.server}/predict", files=files, data=data, timeout=API_TIMEOUT)
            resp.raise_for_status()
            out = resp.json()
        except Exception as e:
            print(f"❌ POST /predict error: {e}")
            out = None
        t_req1 = time.perf_counter()

        if out and out.get("ok"):
            action = out.get("action", [])
            timings = out.get("timings", {})
            # Minimal clean log:
            if (step % args.print_every) == 0:
                print(
                    f"Step {step:04d}  "
                    f"preproc={timings.get('preproc_ms', '?')}ms  "
                    f"infer={timings.get('infer_ms', '?')}ms  "
                    f"total={timings.get('total_ms', '?')}ms  "
                    f"net={(t_req1 - t_req0)*1000:.1f}ms  "
                    f"act={np.array(action, dtype=float)[:5].round(2).tolist()}..."
                )
        else:
            if (step % args.print_every) == 0:
                print(f"Step {step:04d}  no response / error")

        step += 1
        since_print += 1

        # pacing
        t1 = time.perf_counter()
        dt = t1 - t0
        if dt < period:
            time.sleep(period - dt)

        # show achieved loop rate periodically
        if (step % args.print_every) == 0:
            now = time.perf_counter()
            elapsed = now - last_print
            hz = since_print / max(1e-6, elapsed)
            print(f"📈 loop ~ {hz:.2f} Hz over last {since_print} steps")
            last_print = now
            since_print = 0

    # cleanup
    if grip_cap: grip_cap.release()
    if ext_cap: ext_cap.release()
    print("✅ Done.")

if __name__ == "__main__":
    main()
