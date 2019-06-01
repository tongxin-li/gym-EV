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

  def __init__(self, n_EVs=54, n_levels=10, max_energy=20):
    # Parameter for reward function
    self.alpha = 0
    self.beta = 5
    self.gamma = 1
    self.signal = None
    self.state = None
    self.n_EVs = n_EVs
    self.n_levels = n_levels
    self._max_episode_steps = 100000
    self.flexibility = 0
    self.penalty = 0
    self.tracking_error = 0
    self.max_energy = max_energy
    self.rate = 6
    # store previous signal for smoothing
    self.signal_buffer = deque(maxlen=5)
    self.smoothing = 0.5
    # store EV charging result
    self.charging_result = []
    self.initial_bat = []
    self.dic_bat = {}

    # Specify the observation space
    lower_bound = np.array([0])
    upper_bound = np.array([24, 70])
    low = np.append(np.tile(lower_bound, self.n_EVs * 2), lower_bound)
    high = np.append(np.tile(upper_bound, self.n_EVs), np.array([self.max_energy]))
    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Specify the action space
    upper_bound = self.rate
    low = np.append(np.tile(lower_bound, self.n_EVs), np.tile(lower_bound, self.n_levels))
    high = np.append(np.tile(upper_bound, self.n_EVs), np.tile(upper_bound, self.n_levels))
    self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Reset time for new episode
    self.time = 0
    self.time_interval = 0.1
    # store data
    self.data = None

  def step(self, action):
    # Update states according to a naive battery model
    # Time advances
    self.time = self.time + self.time_interval
    # Check if a new EV arrives
    for i in range(len(self.data)):
      if self.data[i, 0] > self.time - self.time_interval and self.data[i, 0] <= self.time:
        # Reject if all spots are full
        if np.where(self.state[:, 2] == 0)[0].size == 0:
          continue
        # Add a new active charging station
        else:
          idx = np.where(self.state[:, 2] == 0)[0][0]
          self.state[idx, 0] = self.data[i, 1]
          self.state[idx, 1] = self.data[i, 2]
          self.state[idx, 2] = 1
          self.dic_bat[idx] = self.data[i, 2]

    # Restrict action
    action[np.where(self.state[:,2] == 0)[0]] = 0
    if np.sum(action[:self.n_EVs]) > self.max_energy:
      action[:self.n_EVs] = 1.0 * action[:self.n_EVs] * self.max_energy / np.sum(action[:self.n_EVs])

    # Update remaining time
    time_result = self.state[:, 0] - self.time_interval
    self.state[:, 0] = time_result.clip(min=0)

    # Update battery
    charging_result = self.state[:, 1] - action[:self.n_EVs] * self.time_interval
    # Battery is full
    for item in range(len(charging_result)):
      if charging_result[item] < 0:
        action[item] = self.state[item, 1] / self.time_interval
    self.state[:, 1] = charging_result.clip(min=0)

    self.penalty = 0
    for i in np.nonzero(self.state[:, 2])[0]:
      # The EV has no remaining time
      if self.state[i, 0] == 0:
        # The EV is overdue
        self.charging_result.append(self.state[i,1])
        self.initial_bat.append(self.dic_bat[i])
        if self.state[i, 1] > 0:
          self.penalty = 10 * self.gamma * self.state[i, 1]
        # Deactivate the EV and reset
        self.state[i, :] = 0

      # Use soft penalty
      # else:
      #   penalty = self.gamma * self.state[0, 1] / self.state[i, 0]

    # Select a new tracking signal
    levels = np.linspace(0, self.max_energy, num=self.n_levels)
    # Set signal zero if feedback is allzero
    if not np.any(action[-self.n_levels:]):
      action[-self.n_levels] = 1
      tmp_signal = 0
      # self.signal = choices(levels, weights=action[-self.n_levels:])[0]
      # self.signal = levels[np.argmax(action[-self.n_levels:])]
    else:
      action[-self.n_levels:] = action[-self.n_levels:] / np.sum(action[-self.n_levels:])
      tmp_signal = choices(levels, weights=action[-self.n_levels:])[0]
      # self.signal = levels[np.argmax(action[-self.n_levels:])]
    self.signal = self.smoothing * np.mean(self.signal_buffer) + (1-self.smoothing) * tmp_signal
    self.signal_buffer.append(self.signal)

    # Update rewards
    # Set entropy zero if feedback is allzero
    if not np.any(action[-self.n_levels:]):
      self.flexibility = 0
    else:
      self.flexibility = self.alpha * (stats.entropy(action[-self.n_levels:])) ** 2

    self.tracking_error = self.beta * (np.sum(action[:self.n_EVs]) - self.signal) ** 2
    reward = (self.flexibility - self.tracking_error - self.penalty) / 100

    done = True if self.time >= 24 else False
    obs = np.append(self.state[:, 0:2].flatten(), self.signal)
    info = {}
    refined_act = action
    return obs, reward, done, info, refined_act

  def reset(self, isTrain):
    # Select a random day and restart

    # name = '/Users/tonytiny/Documents/Github/RLScheduling/real/data' + str(day) + '.npy'
    if isTrain:
      day = random.randint(0, 99)
      name = '/Users/chenliang/Workspace/gym-EV_data/real_train/data' + str(day) + '.npy'
    else:
      day = random.randint(0, 21)
      name = '/Users/chenliang/Workspace/gym-EV_data/real_test/data' + str(day) + '.npy'
    # Load data
    data = np.load(name)
    self.data = data
    # Initialize states and time
    self.state = np.zeros([self.n_EVs, 3])
    # Remaining time
    self.state[0, 0] = data[0, 1]
    # SOC
    self.state[0, 1] = data[0, 2]
    # The charging station is activated
    self.state[0, 2] = 1
    # Select initial signal to be zero -- does not matter since time interval is short
    self.signal = 0
    # self.time = np.floor(data[0, 0]*10) / 10.0
    self.time = data[0, 0]
    # signal buffer
    self.signal_buffer.clear()
    self.signal_buffer.append(self.signal)
    self.charging_result = []
    self.initial_bat = []
    self.dic_bat = {}
    self.dic_bat[0] = self.data[0, 2]


    obs = np.append(self.state[:, 0:2].flatten(), self.signal)
    return obs
