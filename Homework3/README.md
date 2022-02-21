# Obstacle avoidance robot simulation
- Name: Aswath Muthuselvam
- UID: 118286204
- Date: 02/20/2022

## Create Catkin workspace:
From any desired directory
```
mkdir ws/src
cd ws/src
```
Add the following in `.bshrc' file or in terminal window:
```bash
source desired-directory/devel/setup.bash
export TURTLEBOT3_MODEL=burger
```

## Download file:
TurtleBot simulation files:
```
sudo apt-get install ros-${ROS_DISTRO}-turtlebot3-gazebo
```
Obstacle avoidance ROS package: \
(Clone into ROS catkin workspace)
```
git clone https://github.com/aswathselvam/obstacle_avoidance.git
```

## Build project
```
catkin_make
```

## Q1. Run Keyboard control:
```
roslaunch turtlebot3_gazebo turtlebot3_house.launch
rosrun obstacle_avoidance teleop.py
```

## Q2. Run Obstacle avoidance:
```
roslaunch obstacle_avoidance obstacle_avoidance.launch
```
