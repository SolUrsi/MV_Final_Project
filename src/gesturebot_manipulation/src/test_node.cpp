#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Transform.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("orientation_tf");

    tf2_ros::TransformBroadcaster br(node);
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = "end_effector_link";
    transform.child_frame_id = "test_frame";
    transform.transform.translation.x = 0.0;
    transform.transform.translation.y = 0.0;
    transform.transform.translation.z = 0.0;

    tf2::Quaternion q;
    // q.setRPY(0, 0, M_PI / 2);
    q.setRPY(0, -M_PI / 2, 0);
    transform.transform.rotation = tf2::toMsg(q);

    rclcpp::Rate rate(10);
    while (rclcpp::ok()) {
        transform.header.stamp = node->now();
        br.sendTransform(transform);
        rate.sleep();
    }

    rclcpp::shutdown();
}
