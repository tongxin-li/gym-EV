import numpy as np
from scipy import stats
import math
from decimal import Decimal
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

  def __init__(self, max_ev=54, number_level=10, max_capacity=20,  max_rate=6.6):
    # Parameter for reward function
    self.alpha = 1
    self.beta = 1
    self.gamma = 1
    self.data = None
    self.signal = None
    self.state = None
    self.max_ev = max_ev
    self.number_level = number_level
    self._max_episode_steps = 100000
    self.flexibility = 0
    self.total_flexibility = 0
    self.penalty = 0
    self.tracking_error = 0
    self.max_capacity = max_capacity
    self.max_rate = max_rate
    # store previous signal for smoothing
    self.signal_buffer = deque(maxlen=5)
    self.smoothing = 0
    # store EV charging result
    self.charging_result = []
    self.initial_bat = []
    self.dic_bat = {}
    self.day = None

    # Specify the observation space
    lower_bound = np.array([0])
    upper_bound = np.array([24, 70])
    low = np.append(np.tile(lower_bound, self.max_ev * 2), lower_bound)
    high = np.append(np.tile(upper_bound, self.max_ev), np.array([self.max_capacity]))
    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Specify the action space
    upper_bound = self.max_rate
    low = np.append(np.tile(lower_bound, self.max_ev), np.tile(lower_bound, self.number_level))
    high = np.append(np.tile(upper_bound, self.max_ev), np.tile(upper_bound, self.number_level))
    self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Reset time for new episode
    # Time unit is measured in Hr
    self.time = 0
    self.time_interval = 0.1
    # This decides the granuality of control
    self.control_steps = 1
    self.mark = 0
    # store data
    self.data = None

  def step(self, action):

    # Update states according to a naive battery model
    # Time advances
    self.time = self.time + self.time_interval
    self.mark = self.mark +1
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

    # Project action
    action[np.where(self.state[:,2] == 0)[0]] = 0
    if np.sum(action[:self.max_ev]) > self.max_capacity:
      action[:self.max_ev] = 1.0 * action[:self.max_ev] * self.max_capacity / np.sum(action[:self.max_ev])

    # Update remaining time
    time_result = self.state[:, 0] - self.time_interval
    self.state[:, 0] = time_result.clip(min=0)

    # Update battery
    charging_state = self.state[:, 1] - action[:self.max_ev] * self.time_interval

    # Battery is full
    for item in range(len(charging_state)):
      if charging_state[item] < 0:
        action[item] = self.state[item, 1] / self.time_interval
    self.state[:, 1] = charging_state.clip(min=0)

    self.penalty = 0
    for i in np.nonzero(self.state[:, 2])[0]:
      # The EV has no remaining time
      if self.state[i, 0] == 0:
        # The EV is overdue
        self.charging_result = np.append(self.charging_result, self.state[i,1])
        self.initial_bat.append(self.dic_bat[i])
        if self.state[i, 1] > 0:
          self.penalty = self.gamma * self.state[i, 1]
        # Deactivate the EV and reset
        self.state[i, :] = 0

      # Use soft penalty
      # else:
      #   penalty = self.gamma * self.state[0, 1] / self.state[i, 0]

    # Select a new tracking signal
    if self.mark == self.control_steps:
      self.mark = 0
      levels = np.linspace(0, self.max_capacity, num=self.number_level)
      # Set signal zero if feedback is allzero
      if not np.any(action[self.max_ev:]):
        action[self.max_ev] = 1
        tmp_signal = 0
      else:
        # Soft-selection
        tmp_signal = choices(levels, weights=action[self.max_ev:])[0]
        # Hard-selection
        # tmp_signal = levels[np.argmax(action[self.max_ev:])]
      self.signal = self.smoothing * np.mean(self.signal_buffer) + (1-self.smoothing) * tmp_signal
      self.signal_buffer.append(self.signal)

    # Update rewards
    # Set entropy zero if feedback is allzero
    if not np.any(action[self.max_ev:]):
      self.flexibility = 0
    else:
      self.flexibility = self.alpha * stats.entropy(action[self.max_ev:])
      self.total_flexibility = self.total_flexibility + self.flexibility
    # Compute tracking error
    self.tracking_error = self.beta * (np.sum(action[:self.max_ev]) - self.signal) ** 2
    reward = (self.flexibility - self.tracking_error - self.penalty)
    # Return obs and rewards
    done = True if self.time >= 24 else False
    obs = np.append(self.state[:, 0:2].flatten(), self.signal)
    info = {}
    refined_act = action
    return obs, reward, done, info, refined_act

  def sample_episode(self, isTrain):
    # Sample depends on Train/Test
    if isTrain:
      self.day = random.randint(0, 99)
      name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_train/data' + str(self.day) + '.npy'
    else:
      self.day = random.randint(0, 21)
      name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_test/data' + str(self.day) + '.npy'
      # Load data
    data = np.load(name)
    return self.day, data

  def get_episode_by_time(self, day):
    self.day = day
    # name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_greedy/data' + str(self.day) + '.npy'
    name = '/Users/tonytiny/Documents/Github/gym-EV_data/fake/data' + str(self.day) + '.npy'
    # Load data
    data = np.load(name)
    return self.day, data

  def reset(self, isTrain, day):
    # Select a random day and restart
    # _, self.data = self.sample_episode(isTrain)

    # Select day in an chronological order
    _, self.data = self.get_episode_by_time(day)
    done = 0
    if len(self.data) == 0:
      done = 1
      obs = []
      return obs, done

    # Reset values
    self.signal = None
    self.state = None
    self.flexibility = 0
    self.total_flexibility = 0
    self.penalty = 0
    self.tracking_error = 0
    self.charging_result = []
    self.initial_bat = []
    self.dic_bat = {}
    self.day = day

    # Initialize states and time
    self.state = np.zeros([self.max_ev, 3])
    # Remaining time
    self.state[0, 0] = self.data[0, 1]
    # SOC
    self.state[0, 1] = self.data[0, 2]
    # The charging station is activated
    self.state[0, 2] = 1
    # Select initial signal randomly -- does not matter since time interval is short
    # levels = np.linspace(0, self.max_capacity, num=self.number_level)
    # self.signal = choices(levels)[0]
    # Select initial signal as 0
    self.signal = 0
    # self.time = np.floor(data[0, 0]*10) / 10.0
    self.time = self.data[0, 0]
    # signal buffer
    self.signal_buffer.clear()
    self.signal_buffer.append(self.signal)
    self.charging_result = []
    self.initial_bat = []
    self.dic_bat = {}
    self.dic_bat[0] = self.data[0, 2]
    obs = np.append(self.state[:, 0:2].flatten(), self.signal)
    return obs, done