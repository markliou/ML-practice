#!/usr/bin/python

#### note ####
# reference: https://pylessons.com/BipedalWalker-v3-PPO/
# install: pip install gym["box2d"]
#############

import tensorflow as tf 
import numpy as np
import gym

env = gym.make("BipedalWalker-v3")
# print(env.action_space.shape)


while(1):
    env.reset()
    epi_running = True

    while(epi_running == True):
        env.render()
        # give a random action
        action = np.random.uniform(-1., 1., size=env.action_space.shape[0])
        # print(action)
        next_state, reward, done, info = env.step(action)
        print(reward)
        
        if done:
            epi_running = False