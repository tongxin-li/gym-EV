import numpy as np
from scipy import stats
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding

# EV data management
import gym_EV.envs.data_collection as data_collection# Get EV Charging Data
import pymongo
import bson
from datetime import datetime, timedelta

# RL packages
import random  # Handling random number generation
from random import choices
from collections import deque  # Ordered collection with ends

class Action:

  def __init__(self, charging_rate = None, feedback = None):
    self.charging_rate = charging_rate
    self.feedback = feedback
  def set_value(self, charging_rate, feedback):
    self.charging_rate = charging_rate
    self.feedback = feedback

  def random_sample(self):
    self.charging_rate = np.random.rand(54)
    self.feedback = np.random.rand(100)



class EVEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    # Parameter for reward function
    self.alpha = 1
    self.beta = 1
    self.gamma = 1
    self.signal = None
    self.state = None
    # self.feedback = None
    self.time = 0
    self.time_interval = 0.1
    # store data
    self.data = None

  def step(self, action):
    # Update states according to a naive battery model
    self.time = self.time + self.time_interval
    # Check if a new EV arrives
    for i in range(len(self.data)):
      if self.data[i, 0] > self.time - self.time_interval and self.data[i, 0] <= self.time:
        # Add a new active charging station
        self.state[np.where(self.state[:, 4] == 0)[0][0]] = 1
    self.state[:, 3] = self.state[:, 3] + action.charging_rate * self.time_interval

    # Select a new tracking signal
    levels = np.linspace(0, 1000, num=100)
    self.signal = choices(levels, action.feedback)
    unsatisfactory = 0

    for i in np.nonzero(self.state[:, 4])[0]:
      # The EV is overdue
      if self.time - self.state[i, 1] >= 0:
        penalty = 500 * self.gamma
        # Inactivate the EV
        self.state[i, 4] = 0
      else:
        penalty = self.gamma * self.state[0, 3] / (self.state[i, 1] - self.time)
    unsatisfactory = unsatisfactory + penalty
    reward = self.alpha * stats.entropy(action.feedback) - self.beta * (
          np.sum(action.charging_rate) - self.signal) ** 2 + unsatisfactory

    done = True if self.time >= 24 else False
    obs = self.state
    info = {}

    return obs, reward, done, info

  def reset(self):
    # Select a random day and restart
    day = random.randint(1, 65)
    name = 'data/data' + str(day) + '.npy'
    # Load data
    data = np.load(name)
    self.data = data
    # Initialize states and time
    self.state = np.zeros([54, 5])
    # Arrival and Depature
    self.state[0, 0:2] = data[0, 0:2]
    # SOC
    self.state[0, 3] = data[0, 2]
    # The charging station is activated
    self.state[0, 4] = 1
    # Select initial signal to be zero -- does not matter since time interval is short
    self.signal = 0
    self.time = data[0, 0]

if __name__ == '__main__':
  env = EVEnv()
  env.reset()
  action = Action()
  action.random_sample()
  obs, rew, done, info = env.step(action)
  print(rew)
  print(action)
