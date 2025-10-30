#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <control_msgs/action/gripper_command.hpp>
#include <control_msgs/msg/joint_jog.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <gesturebot_msgs/msg/arm_command.hpp>
#include <gesturebot_msgs/msg/drive_command.hpp>
#include <gesturebot_msgs/msg/gripper_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2/LinearMath/Transform.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std::chrono_literals;

std::string pose_to_string(geometry_msgs::msg::PoseStamped& pose) {
    return fmt::format("pose=({}, {}, {}), orientation=({}, {}, {}, {})", pose.pose.position.x,
                       pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x,
                       pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w);
}

std::string pose_to_string(geometry_msgs::msg::Pose& pose) {
    return fmt::format("pose=({}, {}, {}), orientation=({}, {}, {}, {})", pose.position.x,
                       pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y,
                       pose.orientation.z, pose.orientation.w);
}

geometry_msgs::msg::Pose pose_from_transform(tf2::Transform& transform) {
    tf2::Vector3 t = transform.getOrigin();
    tf2::Quaternion q = transform.getRotation();

    geometry_msgs::msg::Pose pose;
    pose.position.x = t.x();
    pose.position.y = t.y();
    pose.position.z = t.z();
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    pose.orientation.w = q.w();

    return pose;
}

constexpr float GRIPPER_MIN_POSITION = -0.010;
constexpr float GRIPPER_MAX_POSITION = 0.019;

constexpr float GRIPPER_POS_DIFF = GRIPPER_MAX_POSITION - GRIPPER_MIN_POSITION;
constexpr float GRIPPER_POS_ONE_PERCENT = GRIPPER_POS_DIFF / 100;

class MotionHandler : public rclcpp::Node {
private:
    using GripperCommand = control_msgs::action::GripperCommand;
    using GripperGoalHandle = rclcpp_action::ClientGoalHandle<GripperCommand>;

    rclcpp::TimerBase::SharedPtr m_InitMoveGroupsTimer;
    rclcpp::TimerBase::SharedPtr m_InitGripperActionServerTimer;

    bool m_MoveGroupsReady;
    bool m_GripperActionServerReady;

    rclcpp_action::Client<GripperCommand>::SharedPtr m_GripperClient;

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> m_ArmGroup;

    rclcpp::Subscription<gesturebot_msgs::msg::ArmCommand>::SharedPtr m_ArmMotionSub;
    rclcpp::Subscription<gesturebot_msgs::msg::DriveCommand>::SharedPtr m_DriveMotionSub;
    rclcpp::Subscription<gesturebot_msgs::msg::GripperCommand>::SharedPtr m_GripperMotionSub;

    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_InfoSub;

public:
    MotionHandler()
        : Node("motion_handler"), m_MoveGroupsReady(false), m_GripperActionServerReady(false) {
        auto cb_group_arm =
            this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions options_arm;
        options_arm.callback_group = cb_group_arm;
        m_ArmMotionSub = this->create_subscription<gesturebot_msgs::msg::ArmCommand>(
            "/arm_motion", rclcpp::SensorDataQoS(),
            std::bind(&MotionHandler::armMotionCallback, this, std::placeholders::_1), options_arm);

        m_DriveMotionSub = this->create_subscription<gesturebot_msgs::msg::DriveCommand>(
            "/drive_motion", 10,
            std::bind(&MotionHandler::driveMotionCallback, this, std::placeholders::_1));

        auto cb_group_gripper =
            this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions options_gripper;
        options_gripper.callback_group = cb_group_gripper;
        m_GripperMotionSub = this->create_subscription<gesturebot_msgs::msg::GripperCommand>(
            "/gripper_motion", rclcpp::SensorDataQoS(),
            std::bind(&MotionHandler::gripperMotionCallback, this, std::placeholders::_1),
            options_gripper);

        m_InfoSub = this->create_subscription<std_msgs::msg::Empty>(
            "/info", 10, std::bind(&MotionHandler::infoCallback, this, std::placeholders::_1));

        m_GripperClient = rclcpp_action::create_client<control_msgs::action::GripperCommand>(
            this, "/gripper_controller/gripper_cmd");

        m_InitMoveGroupsTimer = this->create_wall_timer(
            std::chrono::seconds(1), std::bind(&MotionHandler::initMoveGroups, this));

        m_InitGripperActionServerTimer = this->create_wall_timer(
            std::chrono::seconds(1), std::bind(&MotionHandler::initGripperActionServer, this));
    }

private:
    void infoCallback(const std_msgs::msg::Empty::SharedPtr msg) {
        if (not ready()) return;

        (void)msg;

        // auto joint_values = m_ArmGroup->getCurrentJointValues();
        // joint_values[0] += -M_PI / 4;

        // m_ArmGroup->setJointValueTarget(joint_values);
        // m_ArmGroup->move();
    }

    void armMotionCallback(const gesturebot_msgs::msg::ArmCommand::SharedPtr msg) {
        if (not ready()) return;

        std::vector<geometry_msgs::msg::Pose> waypoints;
        waypoints.push_back(msg->target_pose);

        m_ArmGroup->setMaxVelocityScalingFactor(1.0);
        m_ArmGroup->setMaxAccelerationScalingFactor(1.0);

        // TODO: constraint
        auto current_state = m_ArmGroup->getCurrentState();
        if (current_state) {
            std::string restricted_joint_name = "joint4";
            const double* restricted_joint_value =
                current_state->getJointPositions(restricted_joint_name);
            if (restricted_joint_value) {
                moveit_msgs::msg::Constraints c;
                moveit_msgs::msg::JointConstraint jc;

                jc.joint_name = restricted_joint_name;
                jc.position = *restricted_joint_value;
                jc.tolerance_above = 0.001;
                jc.tolerance_below = 0.001;
                jc.weight = 1.0;
                c.joint_constraints.push_back(jc);
                m_ArmGroup->setPathConstraints(c);
            } else {
                RCLCPP_WARN(get_logger(), "!!!!!!! DIDNT GET JOINT VALUE !!!!!!!!!!!");
            }
        } else {
            RCLCPP_WARN(get_logger(), "!!!!!!! DIDNT GET ROBOT STATE !!!!!!!!!!!");
        }

        moveit_msgs::msg::RobotTrajectory trajectory;
        const double eef_step = 0.1;  // TODO: eef step
        const double jump_threshold = 0.0;
        double fraction =
            m_ArmGroup->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

        if (fraction < 0) {
            RCLCPP_ERROR(
                get_logger(),
                "!!!!!!!!!!!!!!!!!!!!!! COMPUTING CARTESIAN PATH FAILED !!!!!!!!!!!!!!!!!!!!!");
            RCLCPP_ERROR(
                get_logger(),
                "!!!!!!!!!!!!!!!!!!!!!! COMPUTING CARTESIAN PATH FAILED !!!!!!!!!!!!!!!!!!!!!");
            RCLCPP_ERROR(
                get_logger(),
                "!!!!!!!!!!!!!!!!!!!!!! COMPUTING CARTESIAN PATH FAILED !!!!!!!!!!!!!!!!!!!!!");
            RCLCPP_ERROR(
                get_logger(),
                "!!!!!!!!!!!!!!!!!!!!!! COMPUTING CARTESIAN PATH FAILED !!!!!!!!!!!!!!!!!!!!!");
            RCLCPP_ERROR(
                get_logger(),
                "!!!!!!!!!!!!!!!!!!!!!! COMPUTING CARTESIAN PATH FAILED !!!!!!!!!!!!!!!!!!!!!");
            RCLCPP_ERROR(
                get_logger(),
                "!!!!!!!!!!!!!!!!!!!!!! COMPUTING CARTESIAN PATH FAILED !!!!!!!!!!!!!!!!!!!!!");
        } else if (fraction >= 0) {  // TODO: path success threshold
            RCLCPP_INFO(get_logger(), "Path computed with %.2f%% success", fraction * 100.0);
            moveit::planning_interface::MoveGroupInterface::Plan plan;
            plan.trajectory_ = trajectory;
            m_ArmGroup->execute(plan);
        } else {
            RCLCPP_INFO(get_logger(),
                        "Path computed with %.2f%% success !!!!!!!!!!!! Path too bad !!!!!!!!!!!!!",
                        fraction * 100.0);
        }
    }

    void driveMotionCallback(const gesturebot_msgs::msg::DriveCommand::SharedPtr msg) {
        if (not ready()) return;

        (void)msg;
        // auto m_JointPub =
        //     this->create_publisher<control_msgs::msg::JointJog>("/servo_node/delta_joint_cmds",
        //     10);

        // rclcpp::Clock real_clock(RCL_SYSTEM_TIME);

        // control_msgs::msg::JointJog joint_msg;
        // joint_msg.header.frame_id = "link0";
        // joint_msg.header.stamp = real_clock.now();
        // joint_msg.joint_names.push_back("joint1");
        // joint_msg.velocities.push_back(10.0);

        // m_JointPub->publish(joint_msg);
    }

    void gripperMotionCallback(const gesturebot_msgs::msg::GripperCommand::SharedPtr msg) {
        if (not ready()) return;

        float position = GRIPPER_MIN_POSITION + (msg->gripper_percentage * GRIPPER_POS_ONE_PERCENT);

        auto goal_msg = GripperCommand::Goal();
        goal_msg.command.position = position;
        goal_msg.command.max_effort = -1.0;

        m_GripperClient->async_send_goal(goal_msg);
    }

    void initMoveGroups() {
        if (m_ArmGroup) return;

        RCLCPP_INFO(get_logger(), "Initializing MoveIt interfaces...");

        m_ArmGroup = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "arm");

        RCLCPP_INFO(get_logger(), "MoveIt interfaces initialized!");
        m_MoveGroupsReady = true;
        m_InitMoveGroupsTimer->cancel();
    }

    void initGripperActionServer() {
        if (!m_GripperClient->action_server_is_ready()) {
            RCLCPP_INFO(get_logger(), "Waiting for gripper action server...");
        } else {
            RCLCPP_INFO(get_logger(), "Gripper action server connected.");
            m_GripperActionServerReady = true;
            m_InitGripperActionServerTimer->cancel();
        }
    }

    bool ready() { return m_MoveGroupsReady && m_GripperActionServerReady; }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<MotionHandler>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();

    rclcpp::shutdown();

    return 0;
}
