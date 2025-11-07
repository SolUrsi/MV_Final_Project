#!/usr/bin/env python3
"""
Local webcam object detection (no ROS).
- Uses MediaPipe Object Detector (EfficientDet-lite)
- Simple hysteresis + EMA smoothing for a stable box
- Draws guide lines and shows bearing (rad / deg)
- Press 'q' to quit
"""

import cv2, math, time
import numpy as np

# --- MediaPipe Tasks (Object Detector) ---
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

# ================= Configuration =================
MODEL_PATH = 'efficientdet_lite0.tflite'          # put the model file next to this script
ALLOWLIST  = ['bottle']                           # restrict to trash classes
SCORE_ON   = 0.50                                 # hysteresis ON threshold
SCORE_OFF  = 0.35                                 # hysteresis OFF threshold
ALPHA_EMA  = 0.25                                 # 0..1 (higher = faster update)
DEADBAND = 0.01                                   # 1% of width/height
FOV_H_DEG  = 70.0                                 # rough FoV is fine for bearing
GUIDE_BAND = 0.15                                 # guide corridor ±15% around image center
CAM_INDEX  = 0                                    # default webcam
SHOW_NO_TARGET = False                            # hide text when no target
# =================================================

# Create MediaPipe detector (VIDEO mode)
base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
opts = mp_vision.ObjectDetectorOptions(
    base_options=base,
    running_mode=mp_vision.RunningMode.VIDEO,
    category_allowlist=ALLOWLIST,
    score_threshold=min(SCORE_ON, SCORE_OFF)  # allow lower threshold for initial picks
)
detector = mp_vision.ObjectDetector.create_from_options(opts)
ts_ms = 0  # just needs to be monotonically increasing

# State for stable overlay
locked = False
lock_t = 0.0            # last time we had a good detection
smooth = None           # EMA-smoothed (cx, cy, w, h, score)

def ema(prev, cur, alpha):
    """Simple exponential moving average over a tuple."""
    if prev is None: return cur
    return tuple(alpha*c + (1-alpha)*p for p, c in zip(prev, cur))

def should_lock(is_locked, score):
    """Hysteresis: when unlocked, require SCORE_ON; when locked, keep until below SCORE_OFF."""
    return score >= (SCORE_OFF if is_locked else SCORE_ON)

def apply_deadband(prev, cur):
    if prev is None: return cur
    cx, cy, bw, bh, s  = cur
    px, py, pw, ph, ps = prev
    if abs(cx - px) < DEADBAND: cx = px
    if abs(cy - py) < DEADBAND: cy = py
    return (cx, cy, bw, bh, s)

# Open webcam
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit(1)
print("Webcam opened. Press 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed.")
        break

    H, W = frame.shape[:2]

    # Run MediaPipe detection
    ts_ms += 33  # ~30 FPS
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_img, ts_ms)

    # Pick best detection by score
    best = None
    best_s = 0.0
    if result and result.detections:
        for d in result.detections:
            if not d.categories: 
                continue
            s = d.categories[0].score
            if s > best_s:
                best, best_s = d, s

    now = time.time()
    if best is not None:
        bb = best.bounding_box
        cx = (bb.origin_x + bb.width * 0.5) / W
        cy = (bb.origin_y + bb.height * 0.5) / H
        bw = bb.width / W
        bh = bb.height / H
        score = float(best_s)

        # reject very small / noisy detections
        if bw < 0.06 or bh < 0.10:
            best = None  # too small (skip this frame)
        else:
            if should_lock(locked, score):
                locked = True
                lock_t = now
                cur = (cx, cy, bw, bh, score)
                cur = apply_deadband(smooth, cur)
                smooth = ema(smooth, cur, ALPHA_EMA)
            else:
                # If we just dropped below SCORE_OFF, keep the last box for a short grace period (0.8 s)
                if locked and (now - lock_t < 0.8):
                    pass
                else:
                    locked = False
                    smooth = None
    else:
        # No detection: keep the last box briefly, then clear
        if locked and (time.time() - lock_t < 0.8):
            pass
        else:
            locked = False
            smooth = None

    # ---------- Overlay ----------
    # Guide corridor around image center
    xL = int((0.5 - GUIDE_BAND) * W)
    xR = int((0.5 + GUIDE_BAND) * W)
    cv2.line(frame, (xL, 0), (xL, H), (180, 180, 180), 1)
    cv2.line(frame, (xR, 0), (xR, H), (180, 180, 180), 1)
    cv2.line(frame, (W//2, 0), (W//2, H), (220, 220, 220), 1)

    if smooth:
        cx, cy, bw, bh, score = smooth
        x0 = int((cx - bw/2) * W); y0 = int((cy - bh/2) * H)
        x1 = int((cx + bw/2) * W); y1 = int((cy + bh/2) * H)

        # angle (orientation) in degrees from image center (bearing == angle)
        half_fov = math.radians(FOV_H_DEG * 0.5)
        x_norm   = (cx - 0.5) / 0.5
        angle_rad = x_norm * half_fov
        angle_deg = math.degrees(angle_rad)

        # box color = green inside corridor, yellow otherwise
        in_corridor = (xL <= int(cx * W) <= xR)
        color = (0, 255, 0) if in_corridor else (0, 200, 255)

        # draw box (no label text)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

        # multi-line HUD (score, orientation, distance to object (placeholder))
        hud_lines = [
            "Object detected",
            f"  - score: {score:.2f}",
            f"  - orientation: {angle_deg:+.1f} deg",
            f"  - distance: <placeholder> cm",
        ]
        y = 22
        if in_corridor:
            color = (0, 255, 0)
        else:
            color = (0, 200, 255)
        cv2.putText(frame, "Object detected", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for line in hud_lines[1:]:
            y += 18
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    else:
        # nothing printed when no target (unless you want corridor lines only)
        if SHOW_NO_TARGET:
            cv2.putText(frame, "no target", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Webcam Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")