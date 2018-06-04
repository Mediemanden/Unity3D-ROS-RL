# Unity3D-ROS-RL
Using a Unity3D package to connect to ROS topics, robots can be simulated in the game-engine with communication to a Reinforcement Learning Algorithm.

# Prerequisites 
To use the ROS package with Reinforcement Learning, some software is needed to be installed. An Nvidia-graphics card with CUDA compatibility is also needed. 

## Install ROS Kinetic on Ubuntu machine:
Follow instructions: http://wiki.ros.org/kinetic/Installation/Ubuntu 

## Create catkin workspace:
Follow instructions: http://wiki.ros.org/catkin/Tutorials/create_a_workspace 

Use the following command to source the setup.bash file on each terminal startup: 
```
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```
By running the command
```
echo $ROS_PACKAGE_PATH
```
You should now see something like:
```
	/home/username/catkin_ws/src:/opt/ros/kinetic/share
```

## Put file_server package into your catkin workspace

* copy the package into the ~/catkin_ws/src/ folder
* Run catkin_make:
```
roscd
cd .. 
catkin_make
```

## Install dependent packages
* Install ROS-bridge-server
```
sudo apt-get install ros-kinetic-rosbridge-server
```
* Install ROS-control
```
sudo apt-get install ros-kinetic-ros-control ros-kinetic-ros-controllers
```

## Install Tensorflow-GPU, Cuda and Keras
Instructions for installing tensorflow and Cuda: 
https://www.tensorflow.org/install/install_linu

Remember to install the versions of Cuda and CUDNN mentioned in the tutorial.

Suplementary instructions for graphics drivers: http://www.linuxandubuntu.com/home/how-to-install-latest-nvidia-drivers-in-linux 

Having installed tensorflow and pip, install Keras by: 
```
sudo pip install keras
```

To load and save models (this is needed for the RL algorithm), install h5py: 
```
sudo pip install h5py
```

## Download Unity Project
Download the ROSUnityProject to your computer. 

# Getting Started

With all prerequisites done, the Unity project and ROS package are easy to use.

## Set the connection parameters
* Open Unity project.
* Click on the ROSConnector object and change the ROS Bridge Server URL variable of the RosConnector script.
  * The IP-adress should be changed to your current IP adress.
    * Example: ws://169.254.92.54:9090 

## Run reinforcement learning script

To run the reinforcement learning script, use roslaunch: 
```
roslaunch file_server reinforcement_learning.py
```

The RL algorithm is initialised and ready when the output reads
```
--Ready To Run Unity Program--
```

## Run Unity Project
With the RL algorithm running in the terminal, the Unity project can be run by pressing the play button.

The robot is then controlled by the RL package, and moves around according to the action sent to it. 
The terminal should react to the Unity project starting and print information about the reinforcement learning. 
