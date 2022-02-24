# Obstacle avoidance robot simulation
- Name: Aswath Muthuselvam
- UID: 118286204
- Date: 02/20/2022
- Course: ENPM690 - Robot Learning

## Approach:
- We will be using Turtlebot as our robot.
- The Turtlebot is equipped with LIDAR sensor to detect obstacles in its environment.
- We will be using [ROS](https://www.ros.org/) and [Gazebo](http://gazebosim.org/) to control the robot.
- A sample map of a room is loaded in Gazebo environment for the Turtlebot to navigate.

## Download file:
TurtleBot simulation files:
```
sudo apt-get install ros-${ROS_DISTRO}-turtlebot3-gazebo
```

## Create Catkin workspace:
From any `desired directory`:
```
mkdir ws/src
cd ws/src
```
Place this folder `obstacle_avoidance` inside the `ws/src` folder.
Add the following lines in `~/.bashrc` file or in terminal window:
```bash
source <desired-directory>/devel/setup.bash
export TURTLEBOT3_MODEL=burger
```

## Build project:
```
catkin_make
source ~/.bashrc
```

## Q1. Run Keyboard control:
```
roslaunch turtlebot3_gazebo turtlebot3_house.launch
rosrun obstacle_avoidance teleop
```
- [teleop.h](include/teleop.h) and [telop.cpp](src/teleop.cpp) files contain the source code for running keyboard control for the robot.
- Video Demo: Q1.mp4

## Q2. Run Obstacle avoidance:
```
roslaunch obstacle_avoidance obstacle_avoidance.launch
```
- [obstacle_avoidance.h](include/obstacle_avoidance.h) and [obstacle_avoidance.cpp](src/obstacle_avoidance.cpp) files contain the source code for running keyboard control for the robot.
- Video Demo Q2.webm