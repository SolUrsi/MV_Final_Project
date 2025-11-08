
# GestureBot ROS 2 Project 

This project utilizes ROS 2 Humble within a Docker container to host a gesture detection node that controls a simulated or physical TurtleBot3 Waffle Pi robot.

## Prerequisites :shipit:

1. __Docker & Docker Compose__: Must be installed and running on your host machine.

    * Install docker from the official webpage by following their tutorials for your specific OS: [Docker Install](https://docs.docker.com/engine/install/).

2. __Clone project__: Clone the project repository to somewhere on your PC using `git clone https://github.com/corneliusbrandt/MV_Final_Project.git`

3. __Project Structure__: Ensure your local workspace folder (`gesturebot_ws`) is in the same directory as the `Dockerfile` and `docker-compose.yaml` file.
    * This should already be preset as part of cloning the repository

4. __Robot__: For physical testing the TurtleBot3's SBC (Single Board Computer), the Raspberry Pi 4, must be powered on and connected to the same local network as the host machine running this code.
    * Ensure ROS2 Humble is installed on the TurtleBot and that the TurtleBot ROS2 environments match the Docker Compose environments:
        - ROS_DOMAIN_ID=30
        - TURTLEBOT3_MODEL=waffle_pi

## Setting up TurtleBot3 :robot:

To control the physical robot, the TurtleBot3's Single Board Computer (SBC) must be configured to receive the commands published by your container.

1. __Ensure the robot and the application host machine are on the same network!__

2. __Environment setup__: Log into the robot (e.g., via SSH) and ensure the ROS Domain ID matches the container:

```bash
# Set the same communication channel as the Docker container
export ROS_DOMAIN_ID=30

# Set the same TurtleBot3 type as the Docker container
export TURTLEBOT3_MODEL=waffle_pi

# Source the necessary ROS setup files on the robot
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash

# Remember to disable the firewall to allow UDP packet discovery and shipping
sudo ufw disable

```

3. __Launch Robot Bringup__: Run the low-level node to enable motor and sensor communication:

```bash
ros2 launch turtlebot3_manipulation_bringup hardware.launch.py
```

4. __Launch Robot Camerafeed__: Run the ROS2 Camera node to enable the camera topic needed to receive the camerafeed:
```bash
ros2 run camera_ros camera_node --ros-args \
  -p role:=video \
  -p sensor_mode:=1640:1232 \
  -p width:=640 -p height:=480 \
  -p format:=YUYV
```

## Running project instructions ✔️

1. ```bash
    # Move into the cloned repo
    cd MV_Final_Project/
    # Build and start container:
    # -- Building the container is a resource heavy process,
    # -- Terminate all other programs and give the building stage time to finish
    # -- After building it once a majority of packages will be cached, reducing time needed to rebuild
    sudo docker compose build && sudo docker compose up -d

    # Remember to disable the firewall to allow UDP packets to be sent and discovered
    sudo ufw disable
    ```

2. ```bash
    # Ecec into the running docker environment
    sudo docker compose exec gesturebot bash
    ```

## Run commands from project Docker Image 🐳

```bash
# Make sure that you're in ~/gesturebot_ws/ when running these commands
rosdep update
rosdep install --from-paths src -i -y && colcon build --symlink-install
source install/setup.bash
```

This will source the necessary setup scripts to allow the gesture detector to run:

```bash
ros2 run gesture_detector gesture_detector
```

Remember to create a new terminal, exec into it and source the container environment anew.
Then run the Camerafeed Receiver 📷:

```bash
ros2 run camerafeed_receiver camerafeed_receiver.py
```

Enjoy! See the report for gesture mapping to see what gestures allow for robot movement ⬆️!

### Problems❗

Currently it's not possible to run the webcam gesture detector through WSL2 likely due to V4L2 driver issues. Please utilize a Ubuntu 22.04 LTS computer to run the gesture detector! 💻

Make sure to run the robot and the controlling computer over a hotspot network if UDP packets are cleansed from an official Wi-Fi network! 📶


## Finished 🏁

Simply run the following to shut down the container:

```bash
# Make sure to exit the container shell using `ctrl+d` or typing `exit`
sudo docker compose down
```

