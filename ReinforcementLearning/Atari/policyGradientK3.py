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
        self.samplingEpisodes = 10
        self.greedy = .5
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
        self.greedy *= .99
        self.greedy = max(self.greedy, 0.02)

        while (cEpi < self.samplingEpisodes):

            observation = (np.array(observation) - 128.0)/256.0

            # greedy sampling
            agentAction = self.agent(tf.reshape(observation, [1, 210, 160, 3]))
            if np.random.random() < self.greedy:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(agentAction)

            # print(f"action: {action}")

            # interaction with atari
            observation, reward, terminated, truncated, info = self.env.step(
                action)
            observation = (np.array(observation) - 128.0)/256.0
            epiScore += reward

            # if the episode over, the parameters will be reset
            if (terminated == True):
                # show the sampling process information
                print(
                    f'Episode:{cEpi}/{self.samplingEpisodes} score:{epiScore} greedy:{self.greedy}')

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
            actionP = tf.reduce_sum(
                agentAction * tf.stack([tf.one_hot(action, 6)], axis=0))

            if (accumulatedReward != 0.0):
                self.replayBuffer.append(
                    (observation, accumulatedReward, action, actionP.numpy()))
            if (len(self.replayBuffer) > self.bs * 50):
                self.replayBuffer.pop(0)

        # shuffling the replay buffer
        random.shuffle(self.replayBuffer)

    def agent_learning(self):
        # obvStacks, rewardStacks, actionStacks, actionPStacks = zip(
        #     *self.replayBuffer)
        obvStacks = (i[0] for i in self.replayBuffer)
        rewardStacks = (i[1] for i in self.replayBuffer)
        actionStacks = (i[2] for i in self.replayBuffer)
        actionPStacks = (i[3] for i in self.replayBuffer)

        stateDataset = tf.data.Dataset.from_tensor_slices(
            (list(obvStacks), list(rewardStacks), list(actionStacks), list(actionPStacks)))
        stateDataset = stateDataset.batch(self.bs, drop_remainder=True).repeat(8)

        for state in stateDataset:
            # policy gradient training
            obvStack = state[0]
            rewardStack = state[1]
            actionStack = tf.one_hot(tf.stack(state[2], axis=0), 6)
            actionPStack = state[3]

            cLoss = self.update_agent_weights(
                obvStack, rewardStack, actionStack, actionPStack)
            print(f"loss:{cLoss}")

    @tf.function(reduce_retracing=True)
    def update_agent_weights(self, obvStack, rewardStack, actionStack, actionPStack):
        with tf.GradientTape() as grad:
            predicts = self.agent(obvStack)

            # importance sampling
            under = actionPStack
            upper = tf.math.reduce_max(
                predicts * actionStack, axis=-1)
            iSampling = tf.cast(upper/under, tf.float32)

            ce = tf.reduce_sum(
                actionStack * -tf.math.log(predicts + 1e-6), axis=-1)
            policy_ce = tf.reduce_mean(
                tf.cast(rewardStack, tf.float32) * ce * tf.stop_gradient(iSampling))

            gradients = grad.gradient(
                policy_ce, self.agent.trainable_variables)
            self.optimizer.apply(
                gradients, self.agent.trainable_variables)

        return policy_ce


def main():
    ag = agent()
    env = atari_trainer(ag)

    while (1):
        env.sampling()
        env.agent_learning()


if __name__ == "__main__":
    main()
