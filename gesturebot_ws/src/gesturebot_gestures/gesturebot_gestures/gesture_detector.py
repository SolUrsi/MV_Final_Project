from collections import Counter, deque
from enum import Enum

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from gesturebot_msgs.msg import ArmCommand, GripperCommand
from rclpy.node import Node


class Direction(Enum):
    FORWARD = 0
    STOP = 1
    LEFT = 2
    RIGHT = 3
    NONE = 4
    BACKWARD = 5


class GestureDetector(Node):

    def __init__(self):
        super().__init__("gesture_detector")

        self.vel_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.gripper_publisher = self.create_publisher(
            GripperCommand, "gripper_motion", 10
        )
        self.arm_publisher = self.create_publisher(ArmCommand, "arm_motion", 10)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        self.detect_gestures()

    def angle_between(self, v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

    def finger_curl(self, landmarks, ids):
        mcp, pip, tip = [
            np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z]) for i in ids
        ]
        v1 = pip - mcp
        v2 = tip - pip
        angle = self.angle_between(v1, v2)

        straight_deg = 180
        bent_deg = 12
        curl = np.clip((straight_deg - angle) / (straight_deg - bent_deg), 0, 1)
        return curl

    def hand_closure_ratio(self, landmarks):
        fingers = {
            "index": [5, 6, 8],
            "middle": [9, 10, 12],
            "ring": [13, 14, 16],
            "pinky": [17, 18, 20],
        }
        curls = [self.finger_curl(landmarks, ids) for ids in fingers.values()]
        return np.mean(curls)

    def detect_commands(self, hand_landmarks):
        lm = hand_landmarks.landmark
        fingers = []
        direction = Direction.NONE

        if lm[4].x > lm[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

        for finger in [8, 12, 16, 20]:
            if lm[finger].y < lm[finger - 1].y:
                fingers.append(1)
            else:
                fingers.append(0)

        # [thumb, index, middle, ring, pinky]
        if fingers == [0, 1, 0, 0, 0]:
            direction = Direction.FORWARD
        elif fingers == [0, 1, 1, 1, 1]:
            direction = Direction.LEFT
        elif fingers == [1, 0, 0, 0, 0]:
            direction = Direction.RIGHT
        elif fingers == [0, 1, 1, 0, 0]:
            direction = Direction.BACKWARD
        elif fingers == [0, 0, 0, 0, 0]:
            direction = Direction.STOP

        return direction

    def detect_gestures(self):
        PIXEL_TO_M = (1 / 8) * (1 / 100)

        prev_x = None
        prev_y = None
        prev_closure = None

        robot_x = 0
        robot_y = 0
        gripper_closure = 0

        alpha = 0.2
        alpha_gripper = 0.2

        dir_maj_window = deque(maxlen=3)

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                self.get_logger().error("Camera could not be opened!!!")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            center_x, center_y = int((3 / 4) * w), h // 2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            if result.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    handedness = result.multi_handedness[i].classification[0].label
                    score = result.multi_handedness[i].classification[0].score

                    vel_msg = Twist()
                    direction = Direction.STOP

                    if handedness == "Right":
                        gripper_closure = self.hand_closure_ratio(
                            hand_landmarks.landmark
                        )
                        gripper_closure = int(gripper_closure * 100)
                        if prev_closure:
                            gripper_closure = (
                                alpha_gripper * gripper_closure
                                + (1 - alpha_gripper) * prev_closure
                            )
                        gripper_closure = int(gripper_closure)
                        prev_closure = gripper_closure

                        gripper_msg = GripperCommand()
                        gripper_msg.gripper_percentage = gripper_closure
                        self.gripper_publisher.publish(gripper_msg)

                        lm = hand_landmarks.landmark[
                            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ]
                        pixel_x = int(lm.x * w)
                        pixel_y = int(lm.y * h)

                        rel_x = (pixel_x - center_x) * PIXEL_TO_M
                        rel_y = (center_y - pixel_y) * PIXEL_TO_M

                        robot_x = rel_y
                        robot_y = -rel_x

                        if prev_x:
                            robot_x = alpha * robot_x + (1 - alpha) * prev_x
                        prev_x = robot_x

                        if prev_y:
                            robot_y = alpha * robot_y + (1 - alpha) * prev_y
                        prev_y = robot_y

                        arm_msg = ArmCommand()
                        arm_msg.target_pose.position.x = robot_x
                        arm_msg.target_pose.position.y = robot_y
                        arm_msg.target_pose.position.z = 0.2
                        self.arm_publisher.publish(arm_msg)

                        cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 255, 0), -1)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    elif handedness == "Left":
                        direction = self.detect_commands(hand_landmarks)
                        dir_maj_window.append(direction)

                        direction = Counter(dir_maj_window).most_common(1)[0][0]

                        if direction == Direction.FORWARD:
                            vel_msg.linear.x = 0.15
                        elif direction == Direction.STOP:
                            vel_msg.linear.x = 0.0
                        elif direction == Direction.LEFT:
                            vel_msg.angular.z = 0.5
                        elif direction == Direction.RIGHT:
                            vel_msg.angular.z = -0.5
                        elif direction == Direction.BACKWARD:
                            vel_msg.linear.x = -0.15
                    else:
                        pass

                    self.vel_publisher.publish(vel_msg)

                    self.get_logger().info(
                        f"x={robot_x:.2f}m, y={robot_y:.2f}m, closure={gripper_closure}, direction={direction}"
                    )
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

            cv2.imshow("MediaPipe Hands", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    gesture_detector = GestureDetector()

    rclpy.spin(gesture_detector)

    gesture_detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
