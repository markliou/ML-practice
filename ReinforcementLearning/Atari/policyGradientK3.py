import tensorflow as tf
import keras as k
import gym
import random
import numpy as np
from PIL import Image
import time
import random
import threading
import multiprocessing as mp
import timeit


def agent():
    x = k.Input([210, 160, 3])
    conv1 = k.layers.Conv2D(32, [7, 7], strides=[
                            2, 2], padding="SAME", activation=tf.nn.tanh)(x)
    conv2 = k.layers.Conv2D(64, [5, 5], strides=[
                            2, 2], padding="SAME", activation=tf.nn.tanh)(conv1)
    conv3 = k.layers.Conv2D(128, [5, 5], strides=[
                            2, 2], padding="SAME", activation=tf.nn.tanh)(conv2)
    conv4 = k.layers.Conv2D(256, [3, 3], strides=[
                            2, 2], padding="SAME", activation=tf.nn.tanh)(conv3)
    conv5 = k.layers.Conv2D(512, [3, 3], strides=[
                            1, 1], padding="SAME", activation=tf.nn.tanh)(conv4)
    f0 = k.layers.Flatten()(conv5)
    f1 = k.layers.Dense(1024, tf.nn.tanh)(f0)
    f2 = k.layers.Dense(1024, tf.nn.tanh)(f1)
    f3 = k.layers.Dense(1024, tf.nn.tanh)(f2)
    out = k.layers.Dense(6, tf.nn.softmax)(f3)

    return k.Model(x, out)


class atari_trainer():
    def __init__(self, agent, epiNo, cloneAgFunc):
        self.samplingEpisodes = epiNo
        self.env = [gym.make('SpaceInvaders-v4') for i in range(self.samplingEpisodes)]
        # self.env = gym.make('SpaceInvaders-v4', render_mode='human')
        self.gameOverTag = False
        # self.samplingEpisodes = 2
        self.greedy = .2
        self.rewardBufferNo = 20
        self.bs = 32
        self.lr = k.optimizers.schedules.CosineDecay(0.0, 50000, alpha=1e-3, warmup_target=1e-4, warmup_steps=10000)
        self.optimizer = k.mixed_precision.LossScaleOptimizer(k.optimizers.AdamW(self.lr, global_clipnorm=1.))
        # self.optimizer = k.optimizers.AdamW(1e-4, global_clipnorm=1.)
        self.agent = agent
        # self.cloneAg = [cloneAgFunc() for i in range(epiNo
        self.replayBuffer = []
        self.greedyFlag = False

    def pooling_sampling(self):
        self.greedy *= .99
        self.greedy = max(self.greedy, 0.02)
        
        start = timeit.default_timer()
        
        # # multi thread
        # gameEnv = []
        # # fire the threading 
        # for cEpi in range(self.samplingEpisodes):
        #     gameEnv.append(threading.Thread(target = self.sampling, args = (cEpi,)))
        #     # gameEnv.append(mp.Process(target = self.sampling, args = (cEpi,)))
        #     gameEnv[cEpi].start()
            
        # # collecting the threading
        # for cEpi in range(self.samplingEpisodes):
        #     gameEnv[cEpi].join()
        
        # single thread
        for cEpi in range(self.samplingEpisodes):
            self.sampling(cEpi)
            
        print('Epi Sampling Time: ', timeit.default_timer() - start)  

    def sampling(self, eipNo):
        # cloneModel = self.cloneAg[eipNo]
        # cloneModel.set_weights(self.agent.get_weights())
        cloneModel = self.agent
        epiScore = 0
        observation, info = self.env[eipNo].reset()
        observation = (np.array(observation) - 128.0)/128.0
        # the buffer of an action will be counted for self.rewardBufferNo steps
        rewardBuffer = []  
        actionBuffer = []
        actionPBuffer = []
        obvBuffer = []
        
        cLives = info['lives']
        # self.greedy *= .99
        # self.greedy = max(self.greedy, 0.02)
        terminated = False

        while (terminated != True):
            observation = (np.array(observation) - 128.0)/128.0

            # greedy sampling
            agentAction = cloneModel(tf.reshape(observation, [1, 210, 160, 3]))
            if np.random.random() < self.greedy:
                action = self.env[eipNo].action_space.sample()
                self.greedyFlag = True
            else:
                action = np.argmax(agentAction)
                self.greedyFlag = False

            # print(f"action: {action}")

            # interaction with atari
            observation, reward, terminated, truncated, info = self.env[eipNo].step(
                action)
            observation = (np.array(observation) - 128.0)/128.0
            actionP = tf.reduce_sum(
                agentAction * tf.stack([tf.one_hot(action, 6, dtype='bfloat16')], axis=0))
            epiScore += reward

            # if the episode over, the parameters will be reset
            if (terminated == True):
                # show the sampling process information
                print(
                    f'Episode:{eipNo}/{self.samplingEpisodes} score:{epiScore} greedy:{self.greedy}')

                observation, info = self.env[eipNo].reset()
                observation = (np.array(observation) - 128.0)/128.0
                rewardBuffer = []
                actionBuffer = []
                actionPBuffer = []
                obvBuffer = []
                cLives = info['lives']
                epiScore = 0
                continue

            # if the lives reduced, the reward will be minus
            if (info["lives"] < cLives):
                reward = -50
                cLives = info['lives']

            # append the states to the buffer
            
            rewardBuffer.append(reward)
            actionBuffer.append(action)
            actionPBuffer.append(actionP)
            obvBuffer.append(observation)
            
            if (len(rewardBuffer) > self.rewardBufferNo):
                abandentV = rewardBuffer.pop(0)
                del abandentV

            # appending observation into replay buffer. The element limit will be batch size * 100
            # accumulatedReward = np.clip(np.array(rewardBuffer).mean(), -1, 5)
            accumulatedReward = np.array(rewardBuffer).mean()

            if (accumulatedReward != 0.0):
                self.replayBuffer.append((tf.Variable(observation, dtype='bfloat16'),
                                         tf.Variable(accumulatedReward, dtype='bfloat16'),
                                         tf.Variable(action, dtype='int8'),
                                         actionP))
                
            # push more action into replay buffer if the action get a high score
            traceBack = 30
            if (accumulatedReward > 10.0):
                for j in range(4):
                    for i in range(traceBack):
                        self.replayBuffer.append(self.replayBuffer[-1 * traceBack])

            if (len(self.replayBuffer) > self.bs * 100):
                abandentV = self.replayBuffer.pop(0)
                del abandentV

        # shuffling the replay buffer
        random.shuffle(self.replayBuffer)

    def agent_learning(self):
        obvStacks, rewardStacks, actionStacks, actionPStacks = zip(
            *self.replayBuffer)
        #obvStacks = (i[0] for i in self.replayBuffer)
        #rewardStacks = (i[1] for i in self.replayBuffer)
        #actionStacks = (i[2] for i in self.replayBuffer)
        #actionPStacks = (i[3] for i in self.replayBuffer)

        with tf.device('/GPU:1'):
            stateDataset = tf.data.Dataset.from_tensor_slices(
                (list(obvStacks), list(rewardStacks), list(actionStacks), list(actionPStacks)))
            stateDataset = stateDataset.batch(
                self.bs, drop_remainder=True).repeat(1).shuffle(32000)

        for state in stateDataset:
            # policy gradient training
            obvStack = state[0]
            rewardStack = state[1]
            actionStack = tf.one_hot(tf.stack(state[2], axis=0), 6, dtype="bfloat16")
            actionPStack = state[3]

            cLoss = self.update_agent_weights(
                obvStack, rewardStack, actionStack, actionPStack)
            print(f"loss:{cLoss}")
        del obvStacks
        del rewardStacks
        del actionStacks
        del actionPStacks
        del stateDataset
        

    @tf.function(jit_compile=True)
    def update_agent_weights(self, obvStack, rewardStack, actionStack, actionPStack):
        with tf.device('/GPU:0'):
            with tf.GradientTape() as grad:
                predicts = self.agent(obvStack)

                # importance sampling
                under = actionPStack
                upper = tf.math.reduce_max(
                    predicts * actionStack, axis=-1)
                iSampling = tf.cast(upper/under, dtype='bfloat16')
                clippedReward = tf.clip_by_value(tf.cast(rewardStack, dtype='bfloat16') * tf.stop_gradient(iSampling), -1, 5)

                ce = tf.reduce_sum(
                    # actionStack * -tf.math.log(predicts + 1e-6), axis=-1)
                    actionStack * -tf.math.log(tf.clip_by_value(predicts, 1e-6, 1.)), axis=-1)
                policy_ce = tf.reduce_mean(clippedReward * ce )

                gradients = grad.gradient(
                    policy_ce, self.agent.trainable_variables)
                self.optimizer.apply(
                    gradients, self.agent.trainable_variables)

        return policy_ce


def main():
    k.mixed_precision.set_global_policy('mixed_bfloat16')
    ag = agent()
    env = atari_trainer(ag, epiNo=5, cloneAgFunc=agent)

    while (1):
        env.pooling_sampling()
        env.agent_learning()


if __name__ == "__main__":
    main()
