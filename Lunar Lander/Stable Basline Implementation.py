import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# Create the vectorized environment
env = DummyVecEnv([lambda: gym.make("LunarLander-v2", render_mode="human")])
env_old = gym.make("LunarLander-v2", render_mode="human")

# Initialize the model
model = PPO("MlpPolicy", env_old, verbose=1)
model.learn(total_timesteps=1)

# Initialize variables to store rewards
episode_rewards = []
episode_reward = 0
obs = env.reset()

# Run the training loop
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    episode_reward += rewards[0]  # Accumulate rewards
    env.render()

    if done:
        episode_rewards.append(episode_reward)
        episode_reward = 0  # Reset reward for next episode
        obs = env.reset()

env.close()

# Plot the rewards
plt.plot(episode_rewards)
plt.xlabel('steps')
plt.ylabel('Total Reward')
plt.title('Rewards over time')
plt.show()
