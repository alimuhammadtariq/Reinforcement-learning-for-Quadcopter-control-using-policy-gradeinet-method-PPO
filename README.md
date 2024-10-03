# Quadcopter Control Using Reinforcement Learning

## Overview
This project explores reinforcement learning-based approaches to various control problems, particularly focusing on the application of policy gradient algorithms (REINFORCE, PPO, A2C) to OpenAI gym environments like Lunar Lander and Cart Pole, as well as a custom quadcopter physics engine developed using the Pygame library. The aim was to successfully train a quadcopter to hover at an altitude of 300 units using the PPO algorithm, with potential applications in delivery services, search and rescue operations, traffic monitoring, security, and surveillance.

## Features
- **Custom Physics Engine**: Developed using Pygame to simulate the quadcopter environment.
- **Reinforcement Learning Algorithms**: Implemented and tested REINFORCE, PPO, and A2C on different environments.
- **Reward Engineering**: Designed and tested various reward functions to optimize quadcopter hovering.
- **Hypothesis Testing**: Conducted experiments to evaluate the impact of reward scaling, episode length, and boundary constraints on learning and convergence.
- **Applications**: The research can be applied to real-world scenarios like delivery drones, search and rescue operations, and more.

## Implementation Details
The project involved the following key components:
- **Policy Gradient Algorithms**: Applied REINFORCE, PPO, and A2C to solve control problems in OpenAI Gym and custom environments.
- **Quadcopter Physics Engine**: Developed using the Pygame library to simulate real-world quadcopter dynamics.
- **Reward Engineering**: Tested various reward functions with different scaling and boundary conditions to optimize performance:
  - **Reward Function 1-9**: Incremental improvements in design, introducing gradients, boundary constraints, and scaling.
  - **Best Reward Function**: Reward Function 10 was the most successful, achieving stable hovering around 300 units.

## Results
- **Quadcopter Hovering**: Successfully trained the quadcopter to hover at an altitude of 300 units using PPO.
- **Reward Optimization**: The final reward function achieved a stable hovering state with minimal deviations from the target height.
- **Performance Comparison**: Training on Google Colab's GPU outperformed local PC training, showing faster convergence and better results.

## Applications
This research has significant implications in the following areas:
- **Delivery Services**: Autonomous drones for package delivery.
- **Search and Rescue Operations**: Using drones to locate and assist in rescue missions.
- **Traffic Monitoring**: Aerial surveillance to monitor traffic patterns.
- **Security and Surveillance**: Autonomous drones for patrolling and surveillance.
- **Mapping**: Drones for creating high-precision maps.

## Installation and Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/alimuhammadtariq/Reinforcement-learning-for-aerial-Navigation-.git
   cd Reinforcement-learning-for-aerial-Navigation
   pip install -r requirements.txt
   cd "Reinforcement-learning-for-aerial-Navigation-/Quadcopter Hovering"
   python "Quadcopter Hovering/PPO Predictions.py"

## Demo Video

For a visual demonstration of the project, watch the video below:

[![Watch the video](https://img.youtube.com/vi/Ef2fSXwFHAo/0.jpg)](https://youtu.be/9zGyjUetiMY?si=6UBUFrxGp1WFtR-y)



