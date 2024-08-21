import gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from gym import spaces
import numpy as np
import pygame
from PIL import Image, ImageSequence
import seaborn as sns

class QuadcopterEnv(gym.Env):
    def __init__(self, action_space_size=3):
        super(QuadcopterEnv, self).__init__()
        self.action_space = spaces.Discrete(action_space_size)  # Modify action space size
        self.observation_space = spaces.Box(low=np.array([0, -np.inf], dtype=np.float32),
                                            high=np.array([600, np.inf], dtype=np.float32))  # State: [height, velocity]

        pygame.init()
        self.image_path = "Quadcopter.gif"
        self.width, self.screen_height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.screen_height))
        pygame.display.set_caption("Quadcopter Physics Engine")
        self.frame_duration = 100

        resize_dimensions_quadcopter = (80, 80)
        quadcopter = Image.open(self.image_path)
        self.frames = [frame.copy().resize(resize_dimensions_quadcopter, Image.LANCZOS) for frame in
                       ImageSequence.Iterator(quadcopter)]
        self.pygame_frames = [pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode) for frame in self.frames]

        self.frame_index = 0
        self.delta_t = 0.1
        self.step_counter = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = np.array([300.0, 0.0], dtype=np.float32)  # height, y_velocity
        self.step_counter = 0
        return self.current_state, {}

    def step(self, action):
        done = False
        action_values = {0: 0, 1: 30, 2: -10}
        action_value = action_values[action]

        start_y, start_v = self.current_state
        new_v = start_v + (action_value - 9.8) * self.delta_t
        new_y = start_y + new_v * self.delta_t

        self.current_state = np.array([new_y, new_v], dtype=np.float32)

        # Reward logic
        reward = -abs(new_y - 300)/100  # Reward closer to the middle

        # Check if the episode is done
        self.step_counter += 1
        if self.step_counter > 500 or new_y > 590 or new_y < 100:
            done = True

        return self.current_state, reward, done, done, dict()

    def render(self, mode='human'):
        self.frame_index = (self.frame_index + 1) % len(self.pygame_frames)
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.pygame_frames[self.frame_index], (400, self.screen_height - self.current_state[0]))
        pygame.display.flip()
        pygame.time.wait(self.frame_duration)

    def close(self):
        pygame.quit()


# Function to train models and collect rewards
def run_models(model_list, num_episodes=100, num_runs=10):
    all_rewards = {}

    for i in model_list:
        episode_rewards_all_runs = []

        for run in range(num_runs):
            print(f"Running Model {i}, Run {run + 1}/{num_runs}")

            # Adjust action space for model 9 and "10c"
            if i in [9, 10, "10c"]:
                env = QuadcopterEnv(action_space_size=2)
            else:
                env = QuadcopterEnv(action_space_size=3)

            model_path = f"Best Models for each reward/reward {i}"
            model = PPO.load(model_path, env=env)
            episode_rewards = []

            for episode in range(num_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, _ = model.predict(obs)
                    obs, rewards, done, truncated, info = env.step(action.item())
                    episode_reward += rewards

                episode_rewards.append(episode_reward)
            episode_rewards_all_runs.extend(episode_rewards)

        all_rewards[i] = episode_rewards_all_runs

    return all_rewards


# Run models and collect rewards
model_list = [10]
all_rewards = run_models(model_list)

# Plot boxplots for each model
plt.figure(figsize=(12, 6))
sns.boxplot(data=[all_rewards[i] for i in model_list])
plt.xticks(ticks=range(len(model_list)), labels=model_list)
plt.xlabel('Model')
plt.ylabel('Rewards')
plt.title('Comparison 0f Box Plot of all Reward Functions (10 Runs with 100 Episode) ')
plt.savefig(f"Box_Plot_PPO_Quadcopter_Model (comparison).png")
plt.close('all')

# Plot boxplots for individual runs
for i in model_list:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=np.reshape(all_rewards[i], (10, 100)).T)
    plt.xlabel('Run')
    plt.ylabel('Rewards')
    plt.title(f'Box Plot of Reward function {i} Over 10 Runs with 100 episodes each')
    plt.savefig(f"Box Plot of Reward function {i} Over 10 Runs.png")
    plt.close('all')
