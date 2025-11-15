#!/usr/bin/env python3
"""
Remote dual-camera client:
- Captures two frames (external + gripper), makes ONE composite image
- Sends (prompt + image) to /predict
- Receives action vector, applies scaling/safety/adapter
- Commands robot joints
- Measures and reports achieved loop frequency (Hz)

Dependencies:
  pip install opencv-python requests pillow numpy

Local modules expected (already in your repo):
  - improved_soarm_interface.ImprovedSOARMInterface
  - openvla_6motor_adapter.OpenVLA6MotorAdapter
  - soarm_safety_system.SOARMSafetySystem
"""

import argparse
import io
import time
import sys
import requests
import numpy as np
import cv2
from PIL import Image

# ---- Local robot stack (your existing files) ----
from improved_soarm_interface import ImprovedSOARMInterface
from soarm_safety_system import SOARMSafetySystem
from openvla_6motor_adapter import OpenVLA6MotorAdapter


def _open_cam(idx: int, width: int = 1280, height: int = 720, fps: int = 30):
    cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {idx} failed to open")
    return cap


def _composite_bgr(ext_bgr, grip_bgr, comp_h: int) -> np.ndarray:
    """
    Make a single horizontal composite at target height comp_h.
    Keeps aspect ratio of each source, pads to same height, concatenates.
    """
    def _resize_h(img, h):
        h0, w0 = img.shape[:2]
        scale = h / float(h0)
        w = int(w0 * scale)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    ext = _resize_h(ext_bgr, comp_h)
    grip = _resize_h(grip_bgr, comp_h)
    return cv2.hconcat([ext, grip])


def _jpeg_bytes_from_bgr(bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return enc.tobytes()


def _safe_clip(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="Base URL, e.g. http://HOST:8000")
    ap.add_argument("--prompt", default="move to the object")
    ap.add_argument("--gripper-cam", type=int, default=0)
    ap.add_argument("--external-cam", type=int, default=2)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--scale", type=float, default=12.0, help="Overall gain applied after OpenVLA vector")
    ap.add_argument("--comp-h", type=int, default=720, help="Composite image height")
    ap.add_argument("--quiet", action="store_true", help="Minimal per-step logs")
    ap.add_argument("--timeout", type=float, default=2.5, help="Server HTTP timeout (s)")
    args = ap.parse_args()

    # --- Health check (non-fatal) ---
    try:
        r = requests.get(f"{args.server.rstrip('/')}/health", timeout=3.0)
        print(f"Health: {r.json()}")
    except Exception as e:
        print(f"Health: {{'ok': False, 'error': '{e}'}}")

    # --- Cameras ---
    print("📹 Initializing Dual Camera System...")
    ext_cap = _open_cam(args.external-cam if hasattr(args, 'external-cam') else args.external_cam)
    grip_cap = _open_cam(args.gripper-cam if hasattr(args, 'gripper-cam') else args.gripper_cam)
    print("✅ Dual camera system ready!")

    # --- Robot & Safety ---
    safety = SOARMSafetySystem()
    robot = ImprovedSOARMInterface(port="/dev/ttyACM_soarm_leader")  # auto-detected in your stack
    adapter = OpenVLA6MotorAdapter(method="simple_mapping")

    target_dt = 1.0 / max(args.hz, 0.1)
    t_start = time.time()
    t_prev = t_start

    # EMA for achieved Hz
    ema_alpha = 0.1
    ema_dt = None

    step = 0
    moves = 0
    errors = 0

    print(f"🚀 Remote control started for {args.duration:.1f}s @ {args.hz:.1f} Hz")
    print(f"📝 Prompt: {args.prompt}")

    try:
        while True:
            now = time.time()
            if now - t_start >= args.duration:
                break

            # ---- Tick start ----
            loop_t0 = time.time()

            # Capture
            ok1, ext_bgr = ext_cap.read()
            ok2, grip_bgr = grip_cap.read()
            if not (ok1 and ok2):
                if not args.quiet:
                    print("⚠️ Camera read failed; skipping tick")
                time.sleep(min(target_dt, 0.01))
                continue

            comp_bgr = _composite_bgr(ext_bgr, grip_bgr, args.comp_h)
            jpg = _jpeg_bytes_from_bgr(comp_bgr, quality=85)

            # Predict
            try:
                resp = requests.post(
                    f"{args.server.rstrip('/')}/predict",
                    data={"prompt": args.prompt},
                    files={"image": ("comp.jpg", jpg, "image/jpeg")},
                    timeout=args.timeout
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Server {resp.status_code}: {resp.text[:200]}")
                payload = resp.json()
                if not payload.get("ok", False):
                    raise RuntimeError(payload.get("error", "Unknown server error"))

                ova = payload["action"]  # OpenVLA raw action list
            except Exception as e:
                errors += 1
                if not args.quiet:
                    print(f"❌ predict_action failed: {e}")
                time.sleep(min(target_dt, 0.01))
                continue

            # Post-process -> scale + clamp (soft), then safety + adapter
            ova = np.array(ova, dtype=np.float32)

            # Simple global scale then per-joint soft clamp
            scaled = ova * args.scale
            # optional per-index soft clamps (rad-ish), conservative
            limits = [
                (-1.0, 1.0),  # shoulder_pan
                (-1.0, 1.0),  # shoulder_lift
                (-1.0, 1.0),  # elbow_flex
                (-1.0, 1.0),  # wrist_flex
                (-1.2, 1.2),  # wrist_roll
                (-2.0, 2.0),  # gripper (will be re-mapped in your interface)
            ]
            for i in range(min(len(scaled), len(limits))):
                lo, hi = limits[i]
                scaled[i] = _safe_clip(scaled[i], lo, hi)

            # Adapter -> joint space
            joints = adapter.convert(scaled)

            # Safety clamp to hardware limits
            safe_joints = safety.clamp_to_limits(joints)

            # Move robot (non-blocking in your driver)
            moved = robot.move_joints(safe_joints, duration=target_dt)
            moves += 1 if moved else 0

            # ---- Tick end & telemetry ----
            step += 1
            loop_dt = time.time() - loop_t0
            ema_dt = loop_dt if ema_dt is None else (1 - ema_alpha) * ema_dt + ema_alpha * loop_dt
            achieved_hz = (1.0 / ema_dt) if ema_dt and ema_dt > 0 else 0.0

            if step % 5 == 0:
                # compact per-5-step log
                if not args.quiet:
                    j = safe_joints
                    print(
                        f"Step {step:04d}: move={'Y' if moved else 'N'} "
                        f"Hz≈{achieved_hz:4.1f} "
                        f"J=[{j[0]:+0.2f},{j[1]:+0.2f},{j[2]:+0.2f},{j[3]:+0.2f},{j[4]:+0.2f}] "
                        f"G={j[5]:+0.2f}"
                    )

            # pacing
            sleep_left = target_dt - (time.time() - loop_t0)
            if sleep_left > 0:
                time.sleep(sleep_left)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        # Cleanup
        try:
            ext_cap.release()
            grip_cap.release()
        except Exception:
            pass
        try:
            robot.disable_all()
            robot.disconnect()
        except Exception:
            pass

        total = max(time.time() - t_start, 1e-6)
        avg_hz = step / total
        print("📊 Summary")
        print(f"  Steps:       {step}")
        print(f"  Moves:       {moves}")
        print(f"  Errors:      {errors}")
        print(f"  Duration(s): {total:.2f}")
        print(f"  Achieved Hz: {avg_hz:.2f} (EMA≈{achieved_hz:.2f})")


if __name__ == "__main__":
    # Make argparse friendlier when copy/pasting with backslashes
    sys.argv = [a for a in sys.argv if a != "\\"]
    main()
