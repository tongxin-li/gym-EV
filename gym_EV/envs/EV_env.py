import numpy as np
from scipy import stats
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding

# EV data management
# import gym_EV.envs.data_collection as data_collection# Get EV Charging Data
# import pymongo
# import bson
# from datetime import datetime, timedelta

# RL packages
import random  # Handling random number generation
from random import choices
from collections import deque  # Ordered collection with ends



class EVEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    # Parameter for reward function
    self.alpha = 1
    self.beta = 1
    self.gamma = 1
    self.signal = None
    self.state = None

    self.observation_space = spaces.Dict({
      "RemainingTime": spaces.Box(low=0, high=24, shape=(54,)),
      "RemainingEnergy": spaces.Box(low=0, high=24, shape=(54,)),
      "IsActivated": spaces.MultiBinary(54),
      "TrackingSignal": spaces.Box(low=0, high=20, shape=(1,))
    })

    self.action_space = spaces.Dict({
      "ChargingRate": spaces.Box(low=0, high=6, shape=(54,)),
      "Feedback": spaces.Box(low=0, high=1, shape=(10,))
    })

    self.time = 0
    self.time_interval = 0.1
    # store data
    self.data = None

  def step(self, action):
    ## Update states according to a naive battery model
    # Time advances
    self.time = self.time + self.time_interval
    # Check if a new EV arrives
    for i in range(len(self.data)):
      if self.data[i, 0] > self.time - self.time_interval and self.data[i, 0] <= self.time:
        # Add a new active charging station
        self.state[np.where(self.state[:, 2] == 0)[0][0]] = 1

    # Update remaining time
    time_result = self.state[:, 0] - self.time_interval
    self.state[:, 0] = time_result.clip(min=0)

    # Update battery
    charging_result = self.state[:, 1] - action["ChargingRate"] * self.time_interval
    self.state[:, 1] = charging_result.clip(min=0)

    penalty = 0
    for i in np.nonzero(self.state[:, 2])[0]:
      # The EV has no remaining time
      if self.state[i, 0] == 0:
        # The EV is overdue
        if self.state[i, 1] > 0:
          penalty = 10 * self.gamma * self.state[i, 1]
        # Deactivate the EV and reset
        self.state[i, :] = 0
      # else:
      #   penalty = self.gamma * self.state[0, 1] / self.state[i, 0]

    ## Update rewards
    reward = {}
    reward["Flexibility"] = self.alpha * (stats.entropy(action["Feedback"])) ** 2
    reward["TrackingError"] = - self.beta * (
          np.sum(action["ChargingRate"]) - self.signal) ** 2
    reward["OverduePenalty"] = - penalty

    # Select a new tracking signal
    levels = np.linspace(0, 20, num=10)
    self.signal = choices(levels, action["Feedback"])[0]


    done = True if self.time >= 24 else False
    obs = {}
    obs["state"] = self.state.copy()
    obs["signal"] = self.signal
    info = {}
    return obs, reward, done, info

  def reset(self):
    # Select a random day and restart
    day = random.randint(1, 59)
    name = '/Users/tonytiny/Documents/Github/RLScheduling/real/data' + str(day) + '.npy'
    # Load data
    data = np.load(name)
    self.data = data
    # Initialize states and time
    self.state = np.zeros([54, 3])
    # Remaining time
    self.state[0, 0] = data[0, 1]
    # SOC
    self.state[0, 1] = data[0, 2]
    # The charging station is activated
    self.state[0, 2] = 1
    # Select initial signal to be zero -- does not matter since time interval is short
    self.signal = 0
    self.time = data[0, 0]
    obs = {}
    obs["state"] = self.state.copy()
    obs["signal"] = self.signal
    return obs



