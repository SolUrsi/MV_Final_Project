#!/usr/bin/env python3
"""
ROS2 viewer node with real-time object detection and LiDAR distance fusion.
- Subscribes to /camera/image_raw/compressed (JPEG)
- Runs MediaPipe Object Detector (allowlist=['bottle'])
- Stabilizes bounding box (small-box reject + deadband + EMA smoothing)
- Draws guide lines, bounding box, and HUD (score, orientation, LiDAR-based distance)
- Fuses camera detections with /scan data to estimate target range at detected bearing.

Camera Module Information (Raspberry Pi Camera V2.1, Sony IMX219):
- Sensor: 8 MP (3280 x 2464 active pixels)
- Pixel size: 1.12 µm x 1.12 µm
- Sensor size: 3.674 mm (H) x 2.760 mm (V)
- Focal length: 3.04 mm
- Aperture: F/2.0
- Field of View (full sensor readout): 62.2° horizontal, 48.8° vertical
- Video stream modes: 1080p30 (1920x1080), 720p60 (1280x720), 640x480p90
  *These modes are cropped sections of the sensor (reduced field of view).*

  Sources:
  - https://www.waveshare.com/rpi-camera-v2.htm
  - https://www.opensourceinstruments.com/Electronics/Data/IMX219PQ.pdf
"""

# --------------------------------------------------------------------------
# FoV (Field of View) – Reference formulas
#
# (1) Full-Sensor FoV from physical geometry  [Pinhole camera model]
#     FoV_full = 2 * atan( SensorWidth / (2 * FocalLength) )
#         →  SensorWidth  = 3.674 mm  (RPi Camera V2, Sony IMX219)
#         →  FocalLength  = 3.04 mm
#         →  FoV_full_H ≈ 62.2° , FoV_full_V ≈ 48.8°
#
# (2) Effective FoV for cropped (partial) readout
#     FoV_crop = 2 * atan( (Width_crop / Width_full) * tan(FoV_full / 2) )
#
# Notes:
# - Applies only for *central crops* (true sensor cut-out, not scaled).
# - For scaled (downsampled) full-sensor modes → FoV stays constant.
# - Example (RPi V2, 62.2° full sensor):
#       3280 px  → 62.2°   (full sensor)
#       1640 px  → ~33.5°  (≈50% width)
#       1280 px  → ~26.5°  (≈39% width)
#       640 px   → ~13.4°  (≈20% width)
# --------------------------------------------------------------------------

import cv2, math, time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from rclpy.qos import qos_profile_sensor_data

# --- MediaPipe Tasks (Object Detector) ---
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

class CameraFeedReceiver(Node):
    def __init__(self):
        super().__init__('camerafeed_receiver_node')

        # ---------------- Configuration (editable) ----------------
        # --- Object detection config ---
        self.model_path   = 'src/camerafeed_receiver/camerafeed_receiver/efficientdet_lite0.tflite' # model file path
        self.allowlist    = ['bottle']                  # restrict to trash classes (only bottle)
        self.score_on     = 0.50                        # hysteresis ON threshold
        self.score_off    = 0.35                        # hysteresis OFF threshold
        self.alpha_ema    = 0.25                        # EMA smoothing (lower = smoother)
        self.deadband     = 0.01                        # ignore <1% center movement
        self.min_bw       = 0.06                        # reject box width < 6% of frame
        self.min_bh       = 0.10                        # reject box height < 10% of frame
        self.fov_h_deg    = 62.2                        # Raspberry Pi Camera V2 horizontal FOV in degrees
        self.guide_band   = 0.05                        # guide corridor ±5% around center
        self.show_no_target = False                     # keep HUD silent if no target
        self.lock_grace_s = 0.8                         # keep last box briefly on dropouts
        # --- LiDAR fusion config ---
        self.lidar_boresight_deg = 0.0                  # camera - LiDAR alignment offset (deg)
        self.lidar_window_deg    = 3.0                  # window around angle (deg)
        self.lidar_pct           = 30                   # robust percentile
        self.last_scan           = None                 # stores latest LaserScan
        # -----------------------------------------------------------

        # Create MediaPipe detector (VIDEO mode)
        base = mp_python.BaseOptions(model_asset_path=self.model_path)
        opts = mp_vision.ObjectDetectorOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            category_allowlist=self.allowlist,
            score_threshold=min(self.score_on, self.score_off),
        )
        self.detector = mp_vision.ObjectDetector.create_from_options(opts)
        self.ts_ms = 0  # monotonically increasing timestamp for VIDEO mode

        # State for stable overlay
        self.locked = False
        self.lock_t = 0.0            # last time we had a good detection
        self.smooth = None           # EMA-smoothed (cx, cy, w, h, score)

        # Subscribing to camera/image_raw/compressed
        self.subscription = self.create_subscription(
                CompressedImage,
                '/camera/image_raw/compressed',
                self.listener_callback,
                10)
        self.get_logger().info('Camera Feed Receiver Node initialized...')

        # Subscribe to LiDAR scan (for console test)
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data)
        self.get_logger().info('Subscribed to /scan for LiDAR test output...')

    # ---------- tiny helpers (object detection) ----------
    def _ema(self, prev, cur, alpha):
        """Simple exponential moving average over a tuple."""
        if prev is None: 
            return cur
        return tuple(alpha*c + (1-alpha)*p for p, c in zip(prev, cur))

    def _should_lock(self, score):
        """Hysteresis: when unlocked, require SCORE_ON; when locked, keep until below SCORE_OFF."""
        return score >= (self.score_off if self.locked else self.score_on)

    def _apply_deadband(self, prev, cur):
        if prev is None: 
            return cur
        cx, cy, bw, bh, s  = cur
        px, py, pw, ph, ps = prev
        if abs(cx - px) < self.deadband: cx = px
        if abs(cy - py) < self.deadband: cy = py
        return (cx, cy, bw, bh, s)
    
    # ---------- tiny helpers (LiDAR) ----------
    def scan_callback(self, msg: LaserScan):
        # keep latest scan
        self.last_scan = msg
    
    def _distance_for_angle(self, angle_rad):
        """
        Map a camera-based target angle (rad) to LaserScan indices and
        return a robust distance in meters (or None if no valid data).

        Works with LiDAR scans that start at 0 rad (0°) and end near 2*pi rad (359°)
        (TurtleBot3 LDS default, clockwise increment).
        """
        scan = self.last_scan
        if scan is None or not scan.ranges:
            return None

        # --- combine camera angle with boresight offset ---
        theta = angle_rad + math.radians(self.lidar_boresight_deg)

        # Wrap to [0, 2*pi)
        two_pi = 2.0 * math.pi
        theta = theta % two_pi

        # --- read scan geometry ---
        a0  = scan.angle_min               # usually 0.0
        inc = scan.angle_increment         # ≈ +1° in rad
        N   = len(scan.ranges)
        a1  = a0 + inc * N                 # should be close to 2*pi

        # --- convert to scan index ---
        idx = int(round((theta - a0) / inc)) % N

        # --- build local window (± lidar_window_deg) ---
        inc_deg = abs(inc) * 180.0 / math.pi
        half_w = max(0, int(round(abs(self.lidar_window_deg) / max(1e-6, inc_deg))))

        vals = []
        rmin, rmax = scan.range_min, scan.range_max
        for di in range(-half_w, half_w + 1):
            i = (idx + di) % N
            r = scan.ranges[i]
            if math.isfinite(r) and (rmin < r < rmax):
                vals.append(r)

        if not vals:
            return None

        # --- robust percentile distance ---
        vals.sort()
        k = int(round((self.lidar_pct / 100.0) * (len(vals) - 1)))
        return float(vals[k])

    # ---------- main callback ----------
    def listener_callback(self, msg):
        # Converting ROS compressed image message data to OpenCV 
        np_arr = np.frombuffer(msg.data, np.uint8)

        # Decode image data using OpenCV
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn('Failed to decode image.')
            return
        
        H, W = frame.shape[:2]

        # Run MediaPipe detection (VIDEO mode)
        self.ts_ms += 33  # ~30 FPS; needs to increase monotonically
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(mp_img, self.ts_ms)

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
            if bw < self.min_bw or bh < self.min_bh:
                best = None  # too small (skip this frame)
            else:
                if self._should_lock(score):
                    self.locked = True
                    self.lock_t = now
                    cur = (cx, cy, bw, bh, score)
                    cur = self._apply_deadband(self.smooth, cur)
                    self.smooth = self._ema(self.smooth, cur, self.alpha_ema)
                else:
                    # If we just dropped below self.score_off, keep the last box for a short grace period (self.lock_grace_s)
                    if self.locked and (now - self.lock_t < self.lock_grace_s):
                        pass
                    else:
                        self.locked = False
                        self.smooth = None
        else:
            # No detection: keep the last box briefly, then clear
            if self.locked and (time.time() - self.lock_t < self.lock_grace_s):
                pass
            else:
                self.locked = False
                self.smooth = None

        # ---------- Overlay ----------
        # Guide corridor around image center
        xL = int((0.5 - self.guide_band) * W)
        xR = int((0.5 + self.guide_band) * W)
        cv2.line(frame, (xL, 0), (xL, H), (180, 180, 180), 1)
        cv2.line(frame, (xR, 0), (xR, H), (180, 180, 180), 1)
        cv2.line(frame, (W//2, 0), (W//2, H), (220, 220, 220), 1)

        if self.smooth:
            cx, cy, bw, bh, score = self.smooth
            x0 = int((cx - bw/2) * W); y0 = int((cy - bh/2) * H)
            x1 = int((cx + bw/2) * W); y1 = int((cy + bh/2) * H)

            # orientation (deg) from image center (bearing == angle)
            half_fov = math.radians(self.fov_h_deg * 0.5)
            x_norm   = (cx - 0.5) / 0.5
            angle_rad = x_norm * half_fov
            angle_deg = math.degrees(angle_rad)

            # Convert to 0–360° clockwise convention
            angle_deg_cw = (360.0 + angle_deg) % 360.0

            # --- query LiDAR distance at this angle ---
            rng_m = self._distance_for_angle(angle_rad)
            if rng_m is None:
                dist_text = "--- cm"
            else:
                dist_text = f"{rng_m*100:.0f} cm"

            # box color: green inside corridor, yellow otherwise
            in_corridor = (xL <= int(cx * W) <= xR)
            color = (0, 255, 0) if in_corridor else (0, 200, 255)

            # draw box
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

            # --- dark multi-line HUD (always visible)
            hud_color = (0, 0, 240)
            hud_lines = [
                "Object detected",
                f"  - score: {score:.2f}",
                f"  - orientation: {angle_deg_cw:+.1f} deg",
                f"  - distance: {dist_text}",
            ]
            y = 18
            for i, line in enumerate(hud_lines):
                fs = 0.6 if i == 0 else 0.45
                th = 2   if i == 0 else 1
                cv2.putText(frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, hud_color, th)
                y += (22 if i == 0 else 18)
        else:
            if self.show_no_target:
                cv2.putText(frame, "no target", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
                
        cv2.imshow("Camerafeed (Object Detection)", frame)
        cv2.waitKey(1)


def main(args=None):
    # Set up ROS2 execution env
    rclpy.init(args=args)

    camera_feed_receiver = CameraFeedReceiver()

    try:
        # Spin node to process callback function
        rclpy.spin(camera_feed_receiver)
    except KeyboardInterrupt:
        pass

    # Clean up
    camera_feed_receiver.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
