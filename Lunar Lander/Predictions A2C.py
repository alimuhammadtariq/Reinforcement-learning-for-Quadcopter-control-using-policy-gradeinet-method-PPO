#Basic Implementation of stablebaselines from https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/

import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

models_dir = "models/A2C"

env = DummyVecEnv([lambda: gym.make("LunarLander-v2", render_mode="human")])  # continuous: LunarLanderContinuous-v2
env.reset()

model_path = f"{models_dir}/250000.zip"
model = A2C.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)