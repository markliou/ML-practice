import tensorflow as tf
import keras as k
import gym
import random
import numpy as np
from PIL import Image
import time
import random


def agent():
    x = k.Input([210, 160, 3])
    conv1 = k.layers.Conv2D(32, [7, 7], strides=[
                            2, 2], padding="SAME", activation=tf.nn.tanh)(x)
    conv2 = k.layers.Conv2D(64, [5, 5], strides=[
                            2, 2], padding="SAME", activation=tf.nn.tanh)(conv1)
    conv3 = k.layers.Conv2D(128, [5, 5], strides=[
                            2, 2], padding="SAME", activation=tf.nn.tanh)(conv2)
    conv4 = k.layers.Conv2D(256, [5, 5], strides=[
                            2, 2], padding="SAME", activation=tf.nn.tanh)(conv3)
    conv5 = k.layers.Conv2D(512, [5, 5], strides=[
                            2, 2], padding="SAME", activation=tf.nn.tanh)(conv4)
    f0 = k.layers.Flatten()(conv5)
    f1 = k.layers.Dense(1024, tf.nn.tanh)(f0)
    f2 = k.layers.Dense(1024, tf.nn.tanh)(f1)
    f3 = k.layers.Dense(1024, tf.nn.tanh)(f2)
    out = k.layers.Dense(6, tf.nn.softmax)(f3)

    return k.Model(x, out)


class atari_trainer():
    def __init__(self, agent):
        self.env = gym.make('SpaceInvaders-v4')
        # self.env = gym.make('SpaceInvaders-v4', render_mode='human')
        self.gameOverTag = False
        self.samplingEpisodes = 30
        self.greedy = .2
        self.bs = 128
        self.optimizer = k.optimizers.AdamW(1e-4, global_clipnorm=1.)
        self.agent = agent
        self.replayBuffer = []

    def sampling(self):
        cEpi = 0
        epiScore = 0
        observation, info = self.env.reset()
        observation = (np.array(observation) - 128.0)/256.0
        rewardBuffer = []  # the reward of an action will be counted for 30 steps
        cLives = info['lives']

        while (cEpi < self.samplingEpisodes):

            observation = (np.array(observation) - 128.0)/256.0

            # greedy sampling
            if np.random.random() > self.greedy:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.agent(
                    tf.reshape(observation, [1, 210, 160, 3])))

            # interaction with atari
            observation, reward, terminated, truncated, info = self.env.step(
                action)
            observation = (np.array(observation) - 128.0)/256.0
            epiScore += reward

            # if the episode over, the parameters will be reset
            if (terminated == True):
                # show the sampling process informations
                print(
                    f'Episode:{cEpi}/{self.samplingEpisodes} score:{epiScore}')

                cEpi += 1
                observation, info = self.env.reset()
                observation = (np.array(observation) - 128.0)/256.0
                rewardBuffer = []
                cLives = info['lives']
                epiScore = 0
                continue

            # if the lives reduced, the reward will be minus
            if (info["lives"] < cLives):
                reward = -50
                cLives = info['lives']

            # append the reward to the reward buffer
            rewardBuffer.append(reward)
            if (len(rewardBuffer) > 30):
                rewardBuffer.pop(0)

            # appending observation into replay buffer. The element limit will be batch size * 5000
            accumulatedReward = np.array(rewardBuffer).mean()
            if (accumulatedReward != 0.0):
                self.replayBuffer.append(
                    (observation, accumulatedReward, action))
            if (len(rewardBuffer) > self.bs * 5000):
                self.replayBuffer.pop(0)

    def agent_learning(self):
        # shuffling the replay buffer
        random.shuffle(self.replayBuffer)

        bsCounter = 0
        obvStack = []
        rewardStack = []
        actionStack = []
        for i in range(len(self.replayBuffer)):
            # refactor the training data from replay buffer
            bsCounter += 1
            state = self.replayBuffer[i]
            obvStack.append(state[0])
            rewardStack.append(tf.cast(state[1], tf.float32))
            actionStack.append(state[2])

            # policy gradient training
            if (bsCounter == self.bs):
                obvStack = tf.stack(obvStack, axis=0)
                rewardStack = tf.stack(rewardStack, axis=0)
                actionStack = tf.one_hot(tf.stack(actionStack, axis=0), 6)

                @tf.function(reduce_retracing=True)
                def update_agent_weights():
                    with tf.GradientTape() as grad:
                        predicts = self.agent(obvStack)
                        ce = tf.reduce_sum(
                            actionStack * -tf.math.log(predicts + 1e-6), axis=-1)
                        policy_ce = tf.reduce_mean(rewardStack * ce)

                    gradients = grad.gradient(
                        policy_ce, self.agent.trainable_variables)
                    self.optimizer.apply(
                        gradients, self.agent.trainable_variables)

                    return policy_ce

                print(f"loss:{update_agent_weights()}")

                bsCounter = 0
                obvStack = []
                rewardStack = []
                actionStack = []


def main():
    ag = agent()
    env = atari_trainer(ag)

    for i in range(10):
        env.sampling()
        env.agent_learning()


if __name__ == "__main__":
    main()