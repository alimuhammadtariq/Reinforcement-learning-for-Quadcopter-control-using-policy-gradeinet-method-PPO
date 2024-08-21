import matplotlib.pyplot as plt
import numpy as np

# Define the y range from 0 to 600
y_values = np.linspace(0, 600, 600)
#
# Reward 1
reward_1 = np.where((y_values > 300) & (y_values < 350), 100, 0)

# Reward 2-3
reward_2_3 = np.piecewise(y_values,
    [y_values > 400,
     (y_values > 350) & (y_values <= 400),
     (y_values > 300) & (y_values <= 350),
     (y_values > 150) & (y_values <= 300),
     y_values <= 150],
    [-100, -50, 100, -50, -100])

# Reward 4-5
reward_4_5 = np.piecewise(y_values,
    [y_values > 400,
     (y_values > 350) & (y_values <= 400),
     (y_values > 300) & (y_values <= 350),
     (y_values > 150) & (y_values <= 300),
     y_values <= 150],
    [-1, -0.5, 0, -0.5, -1])

# Reward 6
reward_6 = -np.abs(y_values - 300) / 10


# Reward 7
reward_7 = -np.abs(y_values - 300) / 100

# Reward 8
reward_8 = -np.abs(y_values - 300) / 1000


# Plotting the reward functions
plt.figure(figsize=(14, 8))

plt.plot(y_values, reward_1, label='Reward 1')
plt.plot(y_values, reward_2_3, label='Reward 2-3')
plt.plot(y_values, reward_4_5, label='Reward 4-5')
plt.plot(y_values, reward_6, label='Reward 6')
plt.plot(y_values, reward_7, label='Reward 7')
plt.plot(y_values, reward_8, label='Reward 8')

plt.ylim(-10, 10)  # Set y-axis range for better visualization
plt.xlim(0, 600)     # Set x-axis range
plt.title("Reward Functions")
plt.xlabel("y value")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)

plt.show()
