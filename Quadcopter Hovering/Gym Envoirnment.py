import gym
from gym import spaces
import numpy as np
import pygame
from PIL import Image, ImageSequence


class QuadcopterEnv(gym.Env):
    def __init__(self):
        super(QuadcopterEnv, self).__init__()

        self.action_space = spaces.Discrete(3)  # Actions: 0 = no action, 1 = increase thrust, 2 = decrease thrust
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
        action_values = {0: 0, 1: 30, 2: -10}
        action_value = action_values[action]

        start_y, start_v = self.current_state
        new_v = start_v + (action_value - 9.8) * self.delta_t
        new_y = start_y + new_v * self.delta_t

        if new_y > 600 or new_y < 90:
            new_y, new_v = 90, 0

        self.current_state = np.array([new_y, new_v], dtype=np.float32)

        # Reward logic
        # reward = -abs(new_y - 300)  # Reward closer to the middle
        if 290 < new_y < 310:
            reward = 1  # Reward for staying close to the target
        else:
            reward = -1  # Penalty for being far from the target
            # reward=-abs(new_y-300)/10

        # Check if the episode is done
        self.step_counter += 1
        done = self.step_counter >= 5000

        return self.current_state, reward, done, {}

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

# # Example of running the environment
# if __name__ == '__main__':
#     env = gym.make('Quadcopter-v0')
#     state, _ = env.reset()
#     for _ in range(1000):
#         action = env.action_space.sample()  # Take a random action
#         state, reward, done, info = env.step(action)
#         env.render()
#         if done:
#             break
#     env.close()
