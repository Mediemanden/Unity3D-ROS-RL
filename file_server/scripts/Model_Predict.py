#!/usr/bin/env python

# Deep Q-Network reinforcement learning on gym AI MountainCar-v0
# Lukas Rasmus Nielsen & Christoffer Bredo Lillelund
# Multimedia Programming

# Imports:
import sys
# import gym
# import pylab
import os.path

from time import sleep
import random
import numpy as np
from collections import deque
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
# Set start time to calculate full time taken to run the program.
import time
import rospy
import numpy as np
import cv2
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)

# Deep Q-network agent for MountainCar-v0 environment
# Uses neural network for q approximation and experience replay.
class Agent:
    def __init__(self, size_of_states, size_of_actions):

        if os.path.isfile('%s/Models/Prediction_Model.h5' % dir_path):
            self.model = load_model('%s/Models/Prediction_Model.h5' % dir_path)
            print("Model Loaded")
        else:
            print("No Model to load")
            rospy.signal_shutdown("No Model")

    # Epsilon-greedy policy is here used to determine if the action with the highest q-value is to be chosen, or if
    # a random action is to be chosen.
    def get_action(self, state):
            qVal = self.model.predict(state)
            return np.argmax(qVal[0])  # Returns the action which has the highest q-value


def imagecallback(ros_data):
    global agent
    global graph
    global action
    global counter
    global actions
    global image_sub

    #image_sub.unregister()

    if VERBOSE:
        print('received image of type "%s"' % ros_data.format)
        # print("data is {}".format(ros_data.data))

    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.resize(image_np, (80, 60))

    state = image_np.reshape(1, image_np.shape[0], image_np.shape[1], image_np.shape[2])

    with graph.as_default():
        # grab action from the environment in the current state
        action = agent.get_action(state)
        print("Action taken: ", actions[action])
        cmd_vel_pub.publish(actions[action])

    #image_sub = rospy.Subscriber(image_sub_name, CompressedImage, imagecallback, queue_size=1)
    # print(counter)

# Getting the sizes of the observations and actions.
state_size = (60, 80, 3)
action_size = 9

factor = 0.2
factorl = 0.05

twist0 = Twist()
twist0.linear.x = factorl* -1
twist0.angular.z = factor* -1

twist1 = Twist()
twist1.linear.x = factorl* 0
twist1.angular.z = factor* -1

twist2 = Twist()
twist2.linear.x = factorl* 1
twist2.angular.z = factor* -1

twist3 = Twist()
twist3.linear.x = factorl* -1
twist3.angular.z = factor* 0

twist4 = Twist()
twist4.linear.x = factorl* 0
twist4.angular.z = factor* 0

twist5 = Twist()
twist5.linear.x = factorl* 1
twist5.angular.z = factor* 0

twist6 = Twist()
twist6.linear.x = factorl* -1
twist6.angular.z = factor* 1

twist7 = Twist()
twist7.linear.x = factorl* 0
twist7.angular.z = factor* 1

twist8 = Twist()
twist8.linear.x = factorl* 1
twist8.angular.z = factor* 1

actions = [twist0, twist1, twist2, twist3, twist4, twist5, twist6, twist7, twist8]

# Creating the Deep Q-network Agent from the Agent class
agent = Agent(state_size, action_size)

# initializes the scores and episodes arrays (counting scores for each episode) and sets them to nothing.

# Run through each episode, resetting the environment (grabbing initial state) and runnning each step of the environment
# until done is true, then starting a new episode.

VERBOSE = True

# For the real robot
# cmd_vel_name = '/cmd_vel_mux/input/teleop'
# image_sub_name = '/camera/rgb/image_rect_color/compressed'

# For the Unity simulation
cmd_vel_name = 'cmd_vel'
image_sub_name = 'camera/image'

action = None
counter = 1

graph = tf.get_default_graph()

rospy.init_node('Reinforcement_learning')
cmd_vel_pub = rospy.Publisher(cmd_vel_name, Twist, queue_size=100)
global image_sub
image_sub = rospy.Subscriber(image_sub_name, CompressedImage, imagecallback, queue_size=1)

print('--Ready to run Unity program--')

rate = rospy.Rate(20)

while not rospy.is_shutdown():
    rate.sleep()
# END ALL
