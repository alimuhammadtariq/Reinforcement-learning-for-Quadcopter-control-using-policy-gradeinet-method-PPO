#Basic Implementation of stablebaselines from https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/

import gym
from stable_baselines3 import PPO
import os


models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('CartPole-v1')
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000
iters = 0
for i in range(1,1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")