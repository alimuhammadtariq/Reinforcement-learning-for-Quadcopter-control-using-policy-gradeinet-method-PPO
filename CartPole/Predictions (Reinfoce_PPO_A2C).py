
import gym
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


num_episodes=1000

# env = gym.make('CartPole-v1', render_mode="human")
env = gym.make('CartPole-v1', render_mode=None)
policy_network = keras.models.load_model(filepath="Model1" + 'CartPole-v1' + ".h5")
policy_network.summary()

def run_stochastic_policy(policy_network, observation):
    # Reshape observation to (1,num_features)
    observation = observation[np.newaxis,:]
    # Run forward propagation to get softmax probabilities
    action_probabilities = policy_network(observation).numpy().reshape(-1)
    # Select action using a biased sample
    # this will return the index of the action we've sampled
    action = np.random.choice(range(len(action_probabilities)), p=action_probabilities)
    return action

episode_reward_lst=[]

for episode in range(num_episodes):
    observation, info = env.reset(seed=episode)
    episode_reward = 0
    done = False
    steps = 0
    while not (done):
        action = run_stochastic_policy(policy_network, observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = (terminated or terminated)
        episode_reward += reward
        steps += 1
        if steps==500:
            done=True

        if done:
            episode_reward_lst.append(episode_reward)
            print("Episode", episode, "total Reinforce Episode Reward", episode_reward)

        observation = observation_


# env = DummyVecEnv([lambda: gym.make('CartPole-v1', render_mode="human")])
env = DummyVecEnv([lambda: gym.make('CartPole-v1', render_mode= None)])
models_dir = "models/A2C"
model_path = f"{models_dir}/999000.zip"
model = A2C.load(model_path, env=env)
episode_reward_lst_A2C=[]

for episode in range(num_episodes):
    observation = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    while not (done):
        action,_ = model.predict(observation)
        observation_, reward, terminated, info = env.step(action)
        done = (terminated or terminated)
        episode_reward += reward
        steps += 1
        if steps==500:
            done=True

        if done:
            episode_reward_lst_A2C.append(float(episode_reward))
            print("Episode", episode, "total A2C Episode Reward", episode_reward)

        observation = observation_

models_dir = "models/PPO"
model_path = f"{models_dir}/961000.zip"
model = PPO.load(model_path, env=env)
episode_reward_lst_PPO=[]

for episode in range(num_episodes):
    observation = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    while not (done):
        action,_ = model.predict(observation)
        observation_, reward, terminated, info = env.step(action)
        done = (terminated or terminated)
        episode_reward += reward
        steps += 1
        if steps==500:
            done=True

        if done:
            episode_reward_lst_PPO.append(float(episode_reward))
            print("Episode", episode, "total PPO Episode Reward", episode_reward)
        observation = observation_





fig=plt.figure("A2C, PPO, Reinforce Comparison for Cart Pole")
ax=fig.add_subplot(111)
ax.axis([0,episode+1,-600,600])
plt.ylabel('Reward')
plt.xlabel('Episodes')
ax.plot(episode_reward_lst, 'b-', label='Reinforce')
ax.plot(episode_reward_lst_A2C, 'r-', label='A2C')
ax.plot(episode_reward_lst_PPO, 'g-', label='PPO')
ax.set_title("A2C, PPO, Reinforce Comparison")

ax.legend()
plt.draw()
plt.show()

fig = plt.figure("Box plot for Rewards of trained Models (PPO,A2C and Reinforce) Cartpole")
ax = fig.add_subplot(111)
ax.boxplot([episode_reward_lst, episode_reward_lst_A2C, episode_reward_lst_PPO], labels=['REINFORCE', 'A2C', 'PPO'])
ax.set_title("Box plot for Rewards of trained Models (PPO,A2C and Reinforce")
plt.show()