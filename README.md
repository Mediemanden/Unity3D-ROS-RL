# Unity3D-ROS-RL
Using a Unity3D package to connect to ROS topics, robots can be simulated in the game-engine with communication to a Reinforcement Learning Algorithm.

# Prerequisites 
To use the ROS package with Reinforcement Learning, some software is needed to be installed. An Nvidia graphics card with CUDA compatibility is recommended, but not needed. If such graphics card is not available, tensorflow CPU can be used.

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
## Install Unity on Ubuntu machine
A Unity3D installation for Linux can be found [here](https://forum.unity.com/threads/unity-on-linux-release-notes-and-known-issues.350256/).

Optionally, Unity3D can be run on a seperate windows computer with network connection to the Ubuntu machine.


# Getting Started

With all prerequisites done, the Unity project and ROS package are easy to use.

## Put file_server package into your catkin workspace

* Copy the package into the ~/catkin_ws/src/ folder
* Run catkin_make:
```
roscd
cd .. 
catkin_make
```
* Make the python scripts in file_server/scripts/ excecutable:
```
cd ~/catkin_ws/src/file_server/scripts
chmod +x Reinforcement_Learning_Algorithm.py
chmod +x Model_Predict.py
```

## Download Unity Project
Download the ROSUnityProject to the computer with Unity3D installed.

## Set the connection parameters
* Open Unity project.
* Click on the ROSConnector object and change the ROS Bridge Server URL variable of the RosConnector script.
  * The IP-adress should be changed to your current IP adress.
    * Example: ws://169.254.92.54:9090 

## Run reinforcement learning script
Reinforcement learning is done by the Reinforcement_Learning_Algorithm.py script. This script initializes the RL model and handles incoming states, rewards, and outgoing actions.

To run the reinforcement learning script, use roslaunch: 
```
roslaunch file_server reinforcement_learning.launch
```

The RL algorithm is initialised and ready when the output reads
```
--Ready To Run Unity Program--
```

## Run model prediction script
To test a model trained by reinforement learning, a Model Prediction script has been made. This model does not learn from its experiences, but only handles input and provides an outgoing action. 

To run the model prediction script, use roslaunch: 
```
roslaunch file_server model_prediction.launch
```

The prediction algorithm is initialised and ready when the output reads
```
--Ready To Run Unity Program--
```

## Run Unity Project
With a RL algorithm running in the terminal, the Unity project can be run by pressing the play button.

The robot is then controlled by the RL package, and moves around according to the action sent to it. 
The terminal should react to the Unity project starting and print information about the reinforcement learning. 

# Using ROS# for other Unity projects

ROS# is the Unity package used to create a communication with ROS. 

The package is imported in the Unity project of this repository, therefore it works right out of the box. 

To use this package in other Unity projects, use the assets store in Unity to download and import the package called  [**ROS# - ROS-Unity Communication Package**](https://assetstore.unity.com/packages/tools/physics/ros-ros-unity-communication-package-107085)

