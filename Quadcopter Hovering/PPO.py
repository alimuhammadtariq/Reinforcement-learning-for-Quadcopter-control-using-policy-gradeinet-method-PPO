import gym
from stable_baselines3 import PPO
import os
import numpy as np
import pygame
from PIL import Image, ImageSequence
from gym import spaces


# Custom environment class
class QuadcopterEnv(gym.Env):
    def __init__(self):
        super(QuadcopterEnv, self).__init__()

        self.action_space = spaces.Discrete(2)  # Actions: 0 = no action, 1 = increase thrust
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
        self.reward = 0
        return self.current_state, {}

    def step(self, action):
        done = False
        action_values = {0: 0, 1: 30}
        action_value = action_values[action]

        start_y, start_v = self.current_state
        new_v = start_v + (action_value - 9.8) * self.delta_t
        new_y = start_y + new_v * self.delta_t

        self.current_state = np.array([new_y, new_v], dtype=np.float32)

        # Reward logic

        reward = -abs(new_y - 300) / 100


        self.step_counter += 1
        if self.step_counter > 500 or new_y > 590 or new_y < 100:
        # if self.step_counter > 500:
            done = True

        return self.current_state, reward, done, done,  {}

    def render(self, mode='human'):
        self.frame_index = (self.frame_index + 1) % len(self.pygame_frames)

        self.screen.fill((255, 255, 255))
        self.screen.blit(self.pygame_frames[self.frame_index], (400, self.screen_height - self.current_state[0]))
        pygame.display.flip()
        pygame.time.wait(self.frame_duration)

    def close(self)
        pygame.quit()


# Register the custom environment
gym.envs.registration.register(
    id='Quadcopter-v0',
    entry_point='__main__:QuadcopterEnv',
)

# Set up directories for saving models and logs
models_dir = "models/reward 10"
logdir = "logs/reward 10"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Initialize environment and PPO model
env = gym.make('Quadcopter-v0')
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000
for i in range(1, 1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS * i}")



# rough work



# Reward 1
# if 300 < new_y < 350:
#     reward = 100  # Reward for staying close to the target
# else:
#     reward = 0
# if self.step_counter > 500:
#     done = True

# Reward 2-3
# if new_y > 400:
#     reward = -100
# elif 350 < new_y < 400:
#     reward = -50
# elif 300 < new_y < 350:The T
#     reward = 100  # Reward for staying close to the target
# elif 150 < new_y < 300:
#     reward = -50
# elif new_y < 150:
#     reward = -100
# else:
#     reward = 0

# reward 4-5
# if new_y > 400:
#     reward = -1
# elif 350 < new_y < 400:
#     reward = -0.5
# elif 300 < new_y < 350:
#     reward = 0  # Reward for staying close to the target
# elif 150 < new_y < 300:
#     reward = -0.5
# elif new_y < 150:
#     reward = -1
# else:
#     reward = 0

# Reward 6
# reward = -abs(new_y - 300) / 10
# if self.step_counter > 500 or new_y > 590 or new_y < 100:

# Reward 6.1
# reward = -abs(new_y - 300) / 10
# if self.step_counter > 500:


# Reward 7
# reward = -abs(new_y - 300) / 100
# if self.step_counter > 500 or new_y > 590 or new_y < 100: