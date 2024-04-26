import gym
import random
import numpy as np
from PIL import Image

# python -m pip install gym[atari]
# python -m pip install gym[accept-rom-license]

env = gym.make('SpaceInvaders-v0')

observation, info = env.reset()
img = Image.fromarray(observation, "RGB")
img.show()

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample())

    print(f"sample:{env.action_space.sample()} , reward:{reward}")

    if terminated or truncated:
        observation, info = env.reset()

env.close()
