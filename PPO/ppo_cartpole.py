import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

import numpy as np
import tensorflow as tf
import gymnasium as gym
import scipy.signal

import yaml

from ppo_tf import *

# Load config
with open("ppo_config.yaml", "r") as f:
    config = yaml.safe_load(f)

steps_per_epoch = config["steps_per_epoch"]
epochs = config["epochs"]
gamma = config["gamma"]
clip_ratio = config["clip_ratio"]
policy_learning_rate = float(config["policy_learning_rate"])
value_function_learning_rate = float(config["value_function_learning_rate"])
train_policy_iterations = config["train_policy_iterations"]
train_value_iterations = config["train_value_iterations"]
lam = config["lam"]
target_kl = config["target_kl"]
hidden_sizes = tuple(config["hidden_sizes"]) 

render = config["render"] 

# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = gym.make("CartPole-v1")
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize the buffer
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype="float32")
logits = mlp(observation_input, list(hidden_sizes) + [num_actions])
actor = keras.Model(inputs=observation_input, outputs=logits)
value = keras.ops.squeeze(mlp(observation_input, list(hidden_sizes) + [1]), axis=1)
critic = keras.Model(inputs=observation_input, outputs=value)

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

# Initialize the observation, episode return and episode length
observation, _ = env.reset()
episode_return, episode_length = 0, 0