#!/usr/bin/env python

# Deep Q-Network reinforcement learning for Turltebot2 Robot with ROS connection
# Christoffer Bredo Lillelund
# Master Thesis in Medialogy, Aalborg University 2018

# Imports:
import sys
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
from keras.callbacks import Callback
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

# Deep Q-network agent
# Uses neural network for q approximation and experience replay.
class Agent:
    def __init__(self, size_of_states, size_of_actions, epsilons):

        # State and action sizes from class input
        self.state_size = size_of_states
        self.action_size = size_of_actions
        # Setting hyper parameters for the agent
        self.gamma = 0.99  # The discount factor to reduce future reward. Set to 1 means no reduction in future reward value.
        self.lr = 0.0001  # The learning rate of the agent
        if(epsilons == []):
            self.epsilon = 1.0  # Starting epsilon. Sets the percentage of which the agent will perform a random action.
        else:
            self.epsilon = epsilons[len(epsilons)-1]
        self.epsilon_decay = 0.99999  # The epsilon decay is multiplied with the epsilon each episode to decrease it over time
        self.min_epsilon = 0.001  # Sets the minimal epsilon, such that the agent will always explore a little bit.
        self.batch_size = 16  # Batch size for the experience replay
        self.train_start = self.batch_size # Start training after this amount of data has been collected
        self.memory = deque(maxlen=1000000)  # Creates a list (deque) to remember previous data.
        print("Hyper parameters set")

        if os.path.isfile('%s/Models/my_model.h5' % dir_path):
            self.model = load_model('%s/Models/my_model.h5' % dir_path)
            print("Model Loaded")
        else:
            # Creating the models
            self.model = self.build_model()  # model for current state
            print("Model built")
        self.targetState_model = self.build_model()  # model for next state (target)
        print("TargetState Model built")
        # initially sets the parameters and weights of the target model to be the same as the current state model.
        self.update_targetState_model()
        print("TargetState Model updated")

        # approximate Q function using Neural Network

    # state is input and Q Value of each action is output of network

    # function to create the models
    def build_model(self):
        # print("Building Model")
        # Model is made. It is a Sequential model
        model = Sequential()
        # model.add(Dense(256, input_dim=self.state_size, activation='relu'))  # Adds first hidden layer with activation
        #  function Relu to model
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Conv2D(32, (3, 3), activation='relu'))  # new
        model.add(Flatten())

        model.add(Dense(200, activation='relu'))  # Adds second hidden layer to model
        # model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))  # new
        model.add(Dense(self.action_size, activation='linear'))  # add output layer
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))  # sets the loss function and optimizer of the model

        return model

        # Updates the targetState model to have the same weights as the current state model

    def update_targetState_model(self):
        print("Setting weights")
        self.targetState_model.set_weights(self.model.get_weights())  # Sets weights to same as current state model

    # Epsilon-greedy policy is here used to determine if the action with the highest q-value is to be chosen, or if
    # a random action is to be chosen.
    def get_action(self, state):
        print("epsilon: ", self.epsilon)
        if np.random.rand() <= self.epsilon:
            # print("Taking Random Action")
            rand = random.randrange(self.action_size)
            # print("random action is %i" % rand)
            tempQvals.append(0)
            return rand  # Returns a random action
        else:
            # print("Taking best predicted action")
            qVal = self.model.predict(state)
            tempQvals.append(qVal[0, np.argmax(qVal)])
            return np.argmax(qVal[0])  # Returns the action which has the highest q-value

    # Appends the samples from the current state (state, action, reward, next_state, done) to the memory.
    def experience_replay(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # print(self.memory)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay  # Multiplies the epsilon decay such that the chance of random action decreases
            # print("Appended to memory")

    # random sampling from memory. Uses the sample to fit the model with new weights.
    def train_experience_replay(self):
        # print("Training with random sample from memory")
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))

        random_batch_sample = random.sample(self.memory,
                                            batch_size)  # takes a random sample from memory of the size batch_size

        input_batch_state = np.zeros((batch_size, state_size[0], state_size[1], state_size[2]))  # initialises the batch
        input_batch_targetState = np.zeros(
            (batch_size, state_size[0], state_size[1], state_size[2]))  # initialises target batch
        action, reward, done = [], [], []  # initialises action reward and done boolean

        for n in range(self.batch_size):
            input_batch_state[n] = random_batch_sample[n][0]  # inserts states of the random sample batch
            action.append(random_batch_sample[n][1])  # appends actions of the random sample batch to action
            reward.append(random_batch_sample[n][2])  # appends rewards of the random sample batch to reward
            input_batch_targetState[n] = random_batch_sample[n][3]  # inserts next states of the random sample batch
            done.append(random_batch_sample[n][4])  # appends the done status of the random sample batch
        # print("batchvalues set")
        qVal = self.model.predict(input_batch_state)  # predicts the q-values of current state
        target_qVal = self.targetState_model.predict(input_batch_targetState)  # predicts the q-values of next state
        # print("Q-values predicted")
        for n in range(self.batch_size):
            # Provide rewards for each q-value
            if done[n]:
                qVal[n][action[n]] = reward[n]
            else:
                # Value function - if the episode is not done, future rewards are calculated with the discount factor
                qVal[n][action[n]] = reward[n] + self.gamma * np.amax(target_qVal[n])
                # print("Q-values calculated")
                # Fits the model with current states and the corresponding q-values from the batch.
        self.model.fit(input_batch_state, qVal, batch_size=self.batch_size, epochs=1, verbose=1, callbacks=[history])

    # Create the gym environment MountainCar-v0.
    def save_model(self):
        self.model.save('%s/Models/my_model.h5' % dir_path)


class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))

    def on_episode_end(self):
        self.losses = []


def imagecallback(ros_data):
    global i
    global agent
    global scores
    global episodes
    global score
    global state
    global done
    global graph
    global target_state
    global reward
    global action
    global counter
    global actions
    global image_sub
    global times
    global isHit
    global epsilons
    global tempQvals

    image_sub.unregister()

    if VERBOSE:
        print('received image of type "%s"' % ros_data.format)

    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rewardTemp = reward
    reward = 0

    if counter > 199 or rewardTemp > 0:
        done = True
        done_pub.publish("True")


    if target_state is not None:
        with graph.as_default():
            target_state = image_np.reshape(1, image_np.shape[0], image_np.shape[1], image_np.shape[2])
            agent.experience_replay(state, action, rewardTemp, target_state, done)  # Run experience replay class function,
            # appending the state, action, reward, target state and done boolean to the memory
            score += rewardTemp  # summarizes reward

            agent.train_experience_replay()  # runs function to random sample from memory to use for model fitting
            print("score = ", round(score, 1))


    state = image_np.reshape(1, image_np.shape[0], image_np.shape[1], image_np.shape[2])

    with graph.as_default():
        # predict the action to take for the current state
        action = agent.get_action(state)
        cmd_vel_pub.publish(actions[action])
        print(actions[action])
        target_state = state



    if done:
        episodes.append(i)
        scores.append(score)  # Append scores and episodes, for later visualization and system termination
        times.append(time.time() - start_time + supplement_time)
        if counter < 200:
            isHit.append(1)
        else:
            isHit.append(0)
        epsilons.append(agent.epsilon)
        sumLoss.append(sum(history.losses))
        qVals.append(sum(tempQvals))
        tempQvals = []
        if trainingsteps == []:
            trainingsteps.append(counter)
        else:
            trainingsteps.append(trainingsteps[len(trainingsteps)-1] + counter)

        savefile = np.asarray([[episodes], [scores], [isHit], [sumLoss], [epsilons], [trainingsteps], [times], [qVals]])
        savefile = savefile.reshape(8, len(episodes))
        np.savetxt("%s/TestStats/TestStats.csv" % dir_path, np.transpose(savefile), delimiter=",")

        print("Episode:", i, "Score:", round(score, 1), " Time:", round(time.time() - start_time + supplement_time, 1),
              "epsilon:", round(agent.epsilon, 3))  # Print Agent progress
        print("Average score of last ", success_num, " episodes: ", round(np.mean(scores[-min(200, len(scores)):])))
        print("Times ball was hit in the last ", success_num, " episodes: ", np.sum(isHit[-min(200, len(isHit)):]))

        # At the end of every episode update the target model, to the current model.
        agent.update_targetState_model()
        #Save the current model for further use
        agent.save_model()

        if np.sum(isHit[-min(success_num, len(isHit)):]) >= success_num and len(isHit) > 100:
            finish_time = times[len(times)-1]
            print("Achieved goal consecutively after  %s seconds" % finish_time)
            plt.plot(episodes, scores)
            plt.xlabel("Episodes")
            plt.ylabel("Score achieved in episode")
            plt.title("ROS Turtlebot ball navigation \n Deep Q Network Agent")
            plt.show()
            rospy.signal_shutdown("Reinforcement Learning complete")
        done = False
        # Set score to 0, resetting the score for the episode


        score = 0
        i = i + 1
        counter = 0
        history.on_episode_end()


    counter = counter + 1
    image_sub = rospy.Subscriber('camera/image', CompressedImage, imagecallback, queue_size=1)


def rewardcallback(reward_data):
    global reward
    reward = reward + float(reward_data.data)
    if VERBOSE:
        print('received reward: "%i"' % float(reward_data.data))


success_num = 200

# Getting the sizes of the observations and actions.
state_size = (60, 80, 3)

action_size = 9

twist0 = Twist()
twist0.linear.x = -1
twist0.angular.z = -1

twist1 = Twist()
twist1.linear.x = 0
twist1.angular.z = -1

twist2 = Twist()
twist2.linear.x = 1
twist2.angular.z = -1

twist3 = Twist()
twist3.linear.x = -1
twist3.angular.z = 0

twist4 = Twist()
twist4.linear.x = 0
twist4.angular.z = 0

twist5 = Twist()
twist5.linear.x = 1
twist5.angular.z = 0

twist6 = Twist()
twist6.linear.x = -1
twist6.angular.z = 1

twist7 = Twist()
twist7.linear.x = 0
twist7.angular.z = 1

twist8 = Twist()
twist8.linear.x = 1
twist8.angular.z = 1

actions = [twist0, twist1, twist2, twist3, twist4, twist5, twist6, twist7, twist8]

# Creating the Deep Q-network Agent from the Agent class


# initializes the scores and episodes arrays (counting scores for each episode) and sets them to nothing.
if os.path.isfile('%s/TestStats/TestStats.csv' % dir_path):
    statsLoad = np.loadtxt(open('%s/TestStats/TestStats.csv' % dir_path, "r"), delimiter=",")
    episodes = list(statsLoad[:, 0])
    scores = list(statsLoad[:, 1])
    isHit = list(statsLoad[:, 2])
    sumLoss = list(statsLoad[:, 3])
    epsilons = list(statsLoad[:, 4])
    trainingsteps = list(statsLoad[:, 5])
    times = list(statsLoad[:, 6])
    qVals = list(statsLoad[:, 7])
    i = episodes[len(episodes)-1] + 1
    supplement_time = times[len(times)-1]

    print("TestStats Loaded")
else:
    print("No TestStats in folder, setting zero-settings")
    scores, episodes, times, isHit, epsilons, sumLoss, trainingsteps, qVals = [], [], [], [], [], [], [], []
    i = 1
    supplement_time = 0

agent = Agent(state_size, action_size, epsilons)

start_time = time.time()

VERBOSE = False
IMG = False

global history
history = LossHistory()

reward = 0
score = 0
done = False

target_state = None
action = None
counter = 1

graph = tf.get_default_graph()
global tempQvals
tempQvals = []

global image_sub
cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=100)
done_pub = rospy.Publisher('done', String, queue_size=100)
rospy.init_node('Reinforcement_learning')
image_sub = rospy.Subscriber('camera/image', CompressedImage, imagecallback, queue_size=1)
reward_sub = rospy.Subscriber('reward', String, rewardcallback, queue_size=1000)

print('--Ready to run Unity program--')

rate = rospy.Rate(20)

while not rospy.is_shutdown():
    rate.sleep()
# END ALL
