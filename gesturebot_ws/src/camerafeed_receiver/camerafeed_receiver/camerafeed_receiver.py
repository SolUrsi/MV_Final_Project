#!/usr/bin/env python3
"""
ROS2 viewer node with simple object detection overlay.
- Subscribes to /camera/image_raw/compressed (JPEG)
- Runs MediaPipe Object Detector (allowlist=['bottle'])
- Stabilizes bbox (small-box reject + deadband + EMA)
- Draws guide lines, box, and a minimal HUD (score, orientation, distance placeholder)
"""
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
        self.model_path   = 'src/camerafeed_receiver/camerafeed_receiver/efficientdet_lite0.tflite' # model file path
        self.allowlist    = ['bottle']                  # restrict to trash classes (only bottle)
        self.score_on     = 0.50                        # hysteresis ON threshold
        self.score_off    = 0.35                        # hysteresis OFF threshold
        self.alpha_ema    = 0.25                        # EMA smoothing (lower = smoother)
        self.deadband     = 0.01                        # ignore <1% center movement
        self.min_bw       = 0.06                        # reject box width < 6% of frame
        self.min_bh       = 0.10                        # reject box height < 10% of frame
        self.fov_h_deg    = 70.0                        # rough horizontal FoV (without calibration)
        self.guide_band   = 0.15                        # guide corridor ±15% around center
        self.show_no_target = False                     # keep HUD silent if no target
        self.lock_grace_s = 0.8                         # keep last box briefly on dropouts
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

# ==============================================================================================================================

        # Subscribe to LiDAR scan (for console test)
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data)
        self.get_logger().info('Subscribed to /scan for LiDAR test output...')

# ==============================================================================================================================

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
    
# ==============================================================================================================================
    
    # ---------- scan callback ----------
    def scan_callback(self, msg: LaserScan):
        """Simple LiDAR debug printout."""
        N = len(msg.ranges)

        self.get_logger().info(N)
        # find index for 0° (front) if angle_min..angle_max covers −pi..+pi
        i0 = int(round((0.0 - msg.angle_min) / msg.angle_increment))
        if 0 <= i0 < N and math.isfinite(msg.ranges[i0]):
            r_front = msg.ranges[i0]
        else:
            r_front = float('nan')

        self.get_logger().info(
            f"/scan: points={N}  "
            f"angle_min={math.degrees(msg.angle_min):+.1f}°  "
            f"angle_max={math.degrees(msg.angle_max):+.1f}°  "
            f"front≈{r_front:.2f} m"
        )

# ==============================================================================================================================

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

            # box color: green inside corridor, yellow otherwise
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
            y = 18
            for i, line in enumerate(hud_lines):
                fs = 0.6 if i == 0 else 0.45
                th = 2   if i == 0 else 1
                cv2.putText(frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, color, th)
                y += (22 if i == 0 else 18)
        else:
            if self.show_no_target:
                cv2.putText(frame, "no target", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                
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
