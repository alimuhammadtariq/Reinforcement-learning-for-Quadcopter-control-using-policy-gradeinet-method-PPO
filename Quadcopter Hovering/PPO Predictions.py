import gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from gym import spaces
import numpy as np
import pygame
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import seaborn as sns


class QuadcopterEnv(gym.Env):
    def __init__(self):
        super(QuadcopterEnv, self).__init__()

        self.action_space = spaces.Discrete(2)  # Actions: 0 = no action, 1 = increase thrust, 2 = decrease thrust
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


# Register the environment
gym.envs.registration.register(
    id='Quadcopter-v0',
    entry_point='__main__:QuadcopterEnv',
)

# Load the trained model
model_list=["10C"]
for i in model_list:
    model_path = f"Best Models for each reward/reward {i}"

    env = gym.make('Quadcopter-v0')
    model = PPO.load(model_path, env=env)
    episode_reward_lst_PPO=[]

    # Run prediction
    num_episodes = 100
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            env.render()
            action, _states = model.predict(obs)
            obs, rewards, done, truncated, info = env.step(action.item())
            episode_reward += rewards
            print(obs[0])

            if done:
                episode_reward_lst_PPO.append(float(episode_reward))
                print("Episode", episode, "total PPO Episode Reward", episode_reward)


    env.close()



    # fig=plt.figure("PPO for Quadcopter")
    # ax=fig.add_subplot(111)
    # # ax.axis([0,episode+1,-10000,10000])
    # plt.ylabel('Reward')
    # plt.xlabel('Episodes')
    # ax.plot(episode_reward_lst_PPO, 'g-', label='PPO')
    # ax.set_title("PPO for Quadcopter")
    #
    # ax.legend()
    # plt.draw()
    # plt.show()
    #
    # fig = plt.figure("Box plot for Rewards of trained Models (PPO) Quadcopter")
    # ax = fig.add_subplot(111)
    # ax.boxplot([episode_reward_lst_PPO], labels=['PPO'])
    # ax.set_title("Box plot for Rewards of trained Models (PPO")
    # plt.show()0


    # seaborns graphs

    sns.set(style="whitegrid")

    # Plot 1: Reward vs. Episodes using Seaborn
    fig = plt.figure("PPO for Quadcopter")
    ax = fig.add_subplot(111)
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    ax.plot(episode_reward_lst_PPO, 'g-', label='PPO')
    ax.set_title("PPO for Quadcopter")
    ax.legend()

    # Save the figure
    fig.savefig(f"PPO_Quadcopter_Reward_vs_Episodes, reward {i}.png")

    # Plot 2: Box plot for Rewards of trained Models (PPO) Quadcopter using Seaborn
    fig = plt.figure("Box plot for Rewards of trained Models (PPO) Quadcopter")
    ax = fig.add_subplot(111)
    sns.boxplot(data=[episode_reward_lst_PPO], ax=ax)
    ax.set_xticklabels(['PPO'])
    ax.set_title("Box plot for Rewards of trained Models (PPO) Quadcopter")

    # Save the figure
    fig.savefig(f"Box_Plot_PPO_Quadcopter, reward {i}.png")

    # Optional: To ensure all figures are closed
    plt.close('all')