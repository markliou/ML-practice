import gym
import random
import numpy as np
from PIL import Image
import time

# python -m pip install gym[atari]
# python -m pip install gym[accept-rom-license]

# this will show how the game is playing
env = gym.make('SpaceInvaders-v4', render_mode='human')
# env = gym.make('SpaceInvaders-v0')

observation, info = env.reset()
# img = Image.fromarray(observation, "RGB")
# img.show()

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample())

    # time.sleep(1)

    print(
        f"sample:{env.action_space.sample()} , reward:{reward}, live:{info['lives']}")

    if terminated or truncated:
        observation, info = env.reset()

env.close()
