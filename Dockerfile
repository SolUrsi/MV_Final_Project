FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]

ARG USERNAME=robo

ENV ROS_DOMAIN_ID=31
ENV TURTLEBOT3_MODEL=waffle_pi
# ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ENV PROJECT_PATH=gesturebot_ws

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y 
RUN apt upgrade -y

# ROS packages

RUN apt install ros-humble-navigation2 -y
RUN apt install ros-humble-nav2-bringup -y
RUN apt install ros-humble-gazebo-* -y
RUN apt install ros-humble-cartographer -y
RUN apt install ros-humble-cartographer-ros -y

# Misc packages

RUN apt install sudo -y
RUN apt install adduser -y
RUN apt install vim -y
RUN apt install wget -y
RUN apt install xpra -y
RUN apt install x11-utils -y
RUN apt install mesa-utils -y
RUN apt install python3-pip -y
RUN apt install git -y

# User


RUN adduser --disabled-password --gecos "" $USERNAME \
    # Temp removed to check for camera usage
    # && usermod -aG sudo $USERNAME \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && groupadd -f video \
    && usermod -aG video $USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

# turtlebot3

RUN source /opt/ros/humble/setup.bash
RUN mkdir -p ~/turtlebot3_ws/src

RUN cd ~/turtlebot3_ws/src/ \
    && git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git
RUN cd ~/turtlebot3_ws/src/ \ 
    && git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
RUN cd ~/turtlebot3_ws/src/ \ 
    && git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git

RUN sudo apt install python3-colcon-common-extensions

RUN cd ~/turtlebot3_ws \ 
    && source /opt/ros/humble/setup.bash && colcon build --packages-select dynamixel_sdk --symlink-install --parallel-workers 1
RUN cd ~/turtlebot3_ws \ 
    && source /opt/ros/humble/setup.bash && colcon build --packages-select turtlebot3_msgs --symlink-install --parallel-workers 1
RUN cd ~/turtlebot3_ws \ 
    && source /opt/ros/humble/setup.bash && colcon build --packages-select turtlebot3_node --symlink-install --parallel-workers 1

# turtlebot3 simulation Setup

RUN cd ~/turtlebot3_ws/src/ \ 
    && git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git

RUN cd ~/turtlebot3_ws \ 
    && source /opt/ros/humble/setup.bash && colcon build --packages-select turtlebot3_gazebo --symlink-install --parallel-workers 1

RUN cd ~/turtlebot3_ws \ 
    && source /opt/ros/humble/setup.bash && colcon build --symlink-install --parallel-workers 1

# turtlebot3 manipulation Setup

RUN sudo apt install ros-humble-dynamixel-sdk ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-gripper-controllers ros-humble-moveit* -y
RUN cd ~/turtlebot3_ws/src/ \
    && source /opt/ros/humble/setup.bash \
    && git clone -b humble-devel https://github.com/ROBOTIS-GIT/turtlebot3_manipulation.git \
    && cd ~/turtlebot3_ws \
    && colcon build --symlink-install

# .bashrc

RUN echo 'source ~/turtlebot3_ws/install/setup.bash' >> ~/.bashrc
RUN echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc

# 

WORKDIR /home/$USERNAME/$PROJECT_PATH

CMD /bin/bash -c "cd ~/$PROJECT_PATH && rosdep init ; rosdep update && rosdep install --from-paths src -i -y && colcon build --symlink-install ; source install/setup.bash ; /bin/bash -i"





