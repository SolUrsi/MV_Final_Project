
# GestureBot ROS 2 Project 

This project utilizes ROS 2 Humble within a Docker container to host a gesture detection node that controls a simulated or physical TurtleBot3 Waffle Pi robot (hopefully).

## Prerequisites :shipit:

1. __Docker & Docker Compose__: Must be installed and running on your host machine.

    * Install docker from the official webpage by following their tutorials for your specific OS: [Docker Install](https://docs.docker.com/engine/install/).

2. __Clone project__: Clone the project repository to somewhere on your PC using `git clone https://github.com/corneliusbrandt/MV_Final_Project.git`

3. __Project Structure__: Ensure your local workspace folder (`gesturebot_ws`) is in the same directory as the `Dockerfile` and `docker-compose.yaml` file.
    * This should already be preset as part of cloning the repository
4. __Robot__: For physical testing the TurtleBot3 SBC (Single Board Computer), the Raspberry Pi, must be powered on and connected to the same local network as the host machine running this code.
    * Ensure ROS2 Humble is installed on the TurtleBot and that the TurtleBot ROS2 envrionments match the Docker Compose envrionments:
        - ROS_DOMAIN_ID=31
        - TURTLEBOT3_MODEL=waffle_pi

## Setting up TurtleBot3 instructions

To control the physical robot, the TurtleBot3's Single Board Computer (SBC) must be configured to receive the commands published by your container.

1. __Ensure the robot and the application host machine are on the same network!__

2. __Environment setup__: Log into the robot (e.g., via SSH) and ensure the ROS Domain ID matches the container:

```bash
# Set the same communication channel as the Docker container
export ROS_DOMAIN_ID=31

# Set the same TurtleBot3 type as the Docker container
export TURTLEBOT3_MODEL=waffle_pi

# Source the necessary ROS setup files on the robot
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash

# Remember to disable the firewall to allow UDP packet discovery
sudo ufw disable

```

3. __Launch Robot Bringup__: Run the low-level node to enable motor and sensor communication:

```bash
ros2 launch turtlebot3_manipulation_bringup hardware.launch.py
```

## Running project instructions ✔️

1. ```bash
    # cd into the cloned repo
    cd MV_Final_Project/
    # build and start container
    # Building the container is a resource heavy process,
    # Terminate all other programs and give the building stage time to finish
    # After building it once a majority of packages will be cached, reducing time needed
    sudo docker compose build && sudo docker compose up -d

    # Remember to disable the firewall to allow UDP packets to be sent
    sudo ufw disable
    ```
2. ```bash
    sudo docker compose exec gesturebot bash
    ```

### What do these commands do?

1. Builds (or rebuilds) the Docker Image service defined in the Docker Compose file, then pulls the image into the compose commands and starts the container. `-d` makes the container run in a detached mode, making it run in the background instead of taking up the entire terminal. To check the running services use `sudo docker compose ps` or to follow the output logs use `sudo docker compose logs`. 

3. This command is used to run a specific command inside a running container managed by Docker Compose. This activates an interactive shell from within the running container which can run ROS2 commands. 

## Run commands from project Docker Image 🤖 

```bash
# make sure that you're in ~/gesturebot_ws/ when running these commands
rosdep update
rosdep install --from-paths src -i -y && colcon build --symlink-install
source install/setup.bash
```

This will source the necessary setup scripts to allow the gesture detector to run.

```bash
ros2 run gesture_detector gesture_detector
```

This will start the actual python script and (hopefully) open your webcam. To view the robots camerafeed, input the following 📷:

```bash
ros2 run camerafeed_service camerafeed_service.py
```

Enjoy! See the report for the gesture mapping!

### Problems? ✖️

If you're running the application through WSL or other virtual machine or environments, make sure the webcam is exposed to the application!

- ___WSL2___: Make sure to expose the webcam to WSL by following the steps in this link: [USBIPD](https://learn.microsoft.com/en-us/windows/wsl/connect-usb).

Currently it's not possible to run the webcam gesture detector through WSL2 likely due to V4L2 driver issues. Please utilize a Ubuntu 22.04 LTS computer to run the gesture detector! ❗

## Finished?

Simply run the following to shut down the container:

```bash
# make sure to exit the container shell using `ctrl+d` or typing `exit`
sudo docker compose down
```

