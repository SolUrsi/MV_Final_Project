import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class CameraFeedReceiver(Node):
    def __init__(self):
        super().__init__('camerafeed_receiver_node')
        # Subscribing to camera/image_raw/compressed
        self.subscription = self.create_subscription(
                CompressedImage,
                '/camera/image_raw/compressed',
                self.listener_callback,
                10)
        self.subscription
        self.get_logger().info('Camera Feed Receiver Node initialized...')

    def listener_callback(self, msg):
        # Converting ROS compressed image message data to OpenCV 
        np_arr = np.frombuffer(msg.data, np.uint8)

        # Decode image data using OpenCV
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            # Display image in window
            cv2.imshow("Camerafeed", frame)
            cv2.waitkey(1)
        else:
            self.get_logger().warn('Failed to decode image.')


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
