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

from keras import ops


@k.saving.register_keras_serializable(name="RMSNormalization")
class RMSNormalization(k.layers.Layer):
    # from keras-nlp website:
    # https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/src/models/gemma/rms_normalization.py
    # guide: https://keras.io/guides/serialization_and_saving/#config_methods

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    # @classmethod
    # def from_config(cls, config):
    #     # Note that you can also use [`keras.saving.deserialize_keras_object`](/api/models/model_saving_apis/serialization_utils#deserializekerasobject-function) here
    #     return cls(**config)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(input_shape[-1],),
            initializer="zeros",
        )
        self.built = True

    def call(self, x):
        # Always compute normalization in float32.
        x = ops.cast(x, "float32")
        scale = ops.cast(self.scale, "float32")
        var = ops.mean(ops.square(x), axis=-1, keepdims=True)
        normed_inputs = x * ops.reciprocal(ops.sqrt(var + self.epsilon))
        normed_inputs = normed_inputs * (1 + scale)
        return ops.cast(normed_inputs, self.compute_dtype)

def agent():
    x = k.Input([210, 160, 3])
    x = k.layers.LayerNormalization()(x)
    conv1 = k.layers.Conv2D(32, [7, 7], strides=[
                            2, 2], padding="SAME", kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(x)
    # conv1 = k.layers.LayerNormalization()(conv1)
    # conv1 = RMSNormalization()(conv1)
    conv2 = k.layers.Conv2D(64, [5, 5], strides=[
                            2, 2], padding="SAME", kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(conv1)
    # conv2 = k.layers.LayerNormalization()(conv2)
    # conv2 = RMSNormalization()(conv2)
    conv3 = k.layers.Conv2D(128, [5, 5], strides=[
                            2, 2], padding="SAME", kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(conv2)
    # conv3 = k.layers.LayerNormalization()(conv3)
    # conv3 = RMSNormalization()(conv3)
    conv4 = k.layers.Conv2D(256, [3, 3], strides=[
                            2, 2], padding="SAME", kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(conv3)
    # conv4 = k.layers.LayerNormalization()(conv4)
    # conv4 = RMSNormalization()(conv4)
    conv5 = k.layers.Conv2D(512, [3, 3], strides=[
                            1, 1], padding="SAME", kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(conv4)
    f0 = k.layers.Flatten()(conv5)

    # 戰機位置獨立處理
    ag = k.layers.Cropping2D(cropping=((180, 15), (40, 40)))(x)
    conv1_ag = k.layers.Conv2D(32, [7, 7], strides=[
                            2, 2], padding="SAME", kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(ag)
    conv2_ag = k.layers.Conv2D(32, [7, 7], strides=[
                            2, 2], padding="SAME", kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(conv1_ag)
    conv3_ag = k.layers.Conv2D(32, [7, 7], strides=[
                            2, 2], padding="SAME", kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(conv2_ag)
    f0_ag = k.layers.Flatten()(conv3_ag)

    f0 = k.layers.Concatenate()([f0, f0_ag])

    # 防空區獨立處理
    cau = k.layers.Cropping2D(cropping=((140, 50), (40, 40)))(x)
    conv1_cau = k.layers.Conv2D(32, [7, 7], strides=[
                            2, 2], padding="SAME", kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(cau)
    conv2_cau = k.layers.Conv2D(32, [7, 7], strides=[
                            2, 2], padding="SAME", kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(conv1_cau)
    conv3_cau = k.layers.Conv2D(32, [7, 7], strides=[
                            2, 2], padding="SAME", kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(conv2_cau)
    f0_cau = k.layers.Flatten()(conv3_cau)

    f0 = k.layers.Concatenate()([f0, f0_cau])

    # f0 =  k.layers.LayerNormalization(rms_scaling=True)(f0)
    f0 = RMSNormalization()(f0)
    f1 = k.layers.Dense(1024, kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(f0)
    # f1 = k.layers.LayerNormalization(rms_scaling=True)(f1)
    # f1 = RMSNormalization()(f1)
    f2 = k.layers.Dense(512, kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(f1)
    # f2 = k.layers.LayerNormalization(rms_scaling=True)(f2)
    # f2 = RMSNormalization()(f2)
    f3 = k.layers.Dense(256, kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(f2)
    # f3 = k.layers.LayerNormalization(rms_scaling=True)(f3)
    # f3 = RMSNormalization()(f3)
    f4 = k.layers.Dense(128, kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(f3)
    # f4 = k.layers.LayerNormalization(rms_scaling=True)(f4)
    # f4 = RMSNormalization()(f4)
    f5 = k.layers.Dense(64, kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(f4)
    # f5 = k.layers.LayerNormalization(rms_scaling=True)(f5)
    # f5 = RMSNormalization()(f5)
    f6 = k.layers.Dense(32, kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(f5)
    # f6 = k.layers.LayerNormalization(rms_scaling=True)(f6)
    # f6 = RMSNormalization()(f6)
    f7 = k.layers.Dense(16, kernel_regularizer=k.regularizers.L2(1e-4), activation=k.activations.mish)(f6)
    # f7 = k.layers.LayerNormalization(rms_scaling=True)(f7)
    f7 = RMSNormalization()(f7)
    out = k.layers.Dense(6, k.activations.softmax)(f7)

    return k.Model(x, out)

class atari_trainer():
    def __init__(self, agent, epiNo, cloneAgFunc):
        self.samplingEpisodes = epiNo
        self.env = [gym.make('SpaceInvaders-v4') for i in range(self.samplingEpisodes)]
        #self.env = [gym.make('SpaceInvaders-v4', render_mode='human') for i in range(self.samplingEpisodes)]
        self.gameOverTag = False
        self.greedy = .02
        self.rewardBufferNo = 25
        self.bs = 128
        self.lr = k.optimizers.schedules.CosineDecay(0.0, 50000, alpha=1e-3, warmup_target=1e-4, warmup_steps=1000)
        # self.optimizer = k.mixed_precision.LossScaleOptimizer(k.optimizers.AdamW(self.lr, global_clipnorm=1.))
        # self.optimizer = k.mixed_precision.LossScaleOptimizer(k.optimizers.RMSprop(self.lr, rho = .5, global_clipnorm=1.))
        # self.optimizer = k.mixed_precision.LossScaleOptimizer(k.optimizers.SGD(1e-3, momentum=0.9, global_clipnorm=1.))
        self.optimizer = k.mixed_precision.LossScaleOptimizer(k.optimizers.RMSprop(1e-4, rho = .9, global_clipnorm=1.))
        # self.optimizer = k.optimizers.AdamW(1e-4, global_clipnorm=1.)
        self.agent = agent
        # self.cloneAg = [cloneAgFunc() for i in range(epiNo)]
        self.replayBuffer = []
        self.hScoreReplayBuffer = []
        self.greedyFlag = False

    def pooling_sampling(self):
        self.greedy *= .99
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
            if np.random.random(1)[0] > 1.:
                #self.greedy = np.random.random(1)[0]
                self.greedy = .1
            else:
                self.greedy = .02
            #self.greedy = max(self.greedy, 0.02)
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
        localReplayBuffer = []

        cLives = info['lives']
        # self.greedy *= .99
        # self.greedy = max(self.greedy, 0.02)
        terminated = False

        while (terminated != True):
            # greedy sampling for actions
            agentAction = cloneModel(tf.reshape(observation, [1, 210, 160, 3]))
            if np.random.random() < self.greedy:
                action = self.env[eipNo].action_space.sample()
                # action = np.random.randint(4,size=1)[0]
                self.greedyFlag = True
            else:
                action = np.argmax(agentAction)
                self.greedyFlag = False

            # print(f"action: {action}")

            # interaction with atari
            preObservation = observation # for keeping the transition information
            observation, reward, terminated, truncated, info = self.env[eipNo].step(
                action)
            # 把動作歷史放到圖像裡面讓神經網路處理
            observation = (np.array(observation) - 128.0)/128.0 * .6 + preObservation * .2 + np.random.normal(scale=(action / 6),  size=[210, 160, 3])  * .2
            actionP = tf.reduce_sum(
                agentAction * tf.stack([tf.one_hot(action, 6, dtype='float16')], axis=0))
            epiScore += reward

            ## using the transition information
            ### 觀察機體有沒有移動，有移動的分數給高一些

            # 給定位置spectrum，吸引戰機往中央位置移動
            positionSpec = (np.array(
                                    [0 for score in range(40)]+
                                    [(score * 2) / 40. for score in range(40)]+
                                    [(score * 2) / 40. for score in range(39,-1, -1)] +
                                    [0 for score in range(40)]
                                    ))

            positionReward = np.mean((np.abs(observation[180:195,:] - preObservation[180:195,:]) - 0.0078125) * positionSpec.reshape([1,160,1])) # 檢查全域是否有移動
            # reward += np.mean(np.abs(observation[180:195,40:120] - preObservation[180:195,40:120])) # 僅檢查中央部分，如果在中央部分移動就給予高一點的分數
            # reward += np.mean(np.abs(observation[180:195,60:100]) * positionSpec[60:100].reshape([1,40,1])) # 看戰機有沒有落在中央部分。因為背景是黑色，所以就把有顏色的當作戰機
            # positionReward += 5 * np.mean(np.abs(observation[180:195,:]) * positionSpec.reshape([1,160,1])) - 0.302114693102719 # 看戰機有沒有落在中央部分。因為背景是黑色，所以就把有顏色的當作戰機

            # if the episode over, the parameters will be reset
            if (terminated == True):
                # show the sampling process information
                print(
                    f'Episode:{eipNo}/{self.samplingEpisodes} score:{epiScore} greedy:{self.greedy}')

                if epiScore > 400:
                    self.hScoreReplayBuffer += localReplayBuffer

                observation, info = self.env[eipNo].reset()
                observation = (np.array(observation) - 128.0)/128.0
                rewardBuffer = []
                actionBuffer = []
                actionPBuffer = []
                obvBuffer = []
                localReplayBuffer = []
                cLives = info['lives']
                epiScore = 0
                continue

            # if the lives reduced, the reward will be minus
            if (info["lives"] < cLives):
                deadP = .25
                broadcast = 16
                for replayBufferInd in range(broadcast):
                    element = self.replayBuffer[-(replayBufferInd + 1)]
                    self.replayBuffer[-(replayBufferInd + 1)] = (element[0],
                                                           element[1] - ((deadP/broadcast) * (broadcast - replayBufferInd)),
                                                           element[2],
                                                           element[3]
                                                           )
                reward = -(deadP * broadcast)
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
                accumulatedReward = np.array(rewardBuffer).mean() #+ positionReward

                if(True):
                # if (accumulatedReward != 0.0):
                # if (reward != 0.0):
                    localReplayBuffer.append((tf.Variable(obvBuffer[0], dtype='float16'),
                                            tf.Variable(accumulatedReward, dtype='float16'),
                                            tf.Variable(actionBuffer[0], dtype='int8'),
                                            actionPBuffer[0]))
                    self.replayBuffer.append((tf.Variable(obvBuffer.pop(0), dtype='float16'),
                                            tf.Variable(accumulatedReward, dtype='float16'),
                                            tf.Variable(actionBuffer.pop(0), dtype='int8'),
                                            actionPBuffer.pop(0)))


                # # push more action into replay buffer if the action get a high score
                # if (reward >= 50.0):
                #     for j in range(3):
                #         for i in range(self.rewardBufferNo):
                #             self.replayBuffer.append(self.replayBuffer[-1 * self.rewardBufferNo])
                #             if (len(self.replayBuffer) > self.bs * 100):
                #                 abandentV = self.replayBuffer.pop(0)
                #                 del abandentV

                if (len(self.replayBuffer) > self.bs * 100):
                    abandentV = self.replayBuffer.pop(0)
                    del abandentV
                while (len(self.hScoreReplayBuffer) > self.bs * 100):
                    abandentV = self.hScoreReplayBuffer.pop(0)
                    del abandentV

        # shuffling the replay buffer
        random.shuffle(self.replayBuffer)
        random.shuffle(self.hScoreReplayBuffer)

    def agent_learning(self):
        obvStacks, rewardStacks, actionStacks, actionPStacks = zip(
            *(self.replayBuffer))

        with tf.device('/GPU:0'):
            stateDataset = tf.data.Dataset.from_tensor_slices(
                (list(obvStacks), list(rewardStacks), list(actionStacks), list(actionPStacks)))
            stateDataset = stateDataset.batch(
                self.bs, drop_remainder=True).repeat(8).shuffle(32000)

        for state in stateDataset:
            # policy gradient training
            obvStack = state[0]
            rewardStack = state[1]
            actionStack = tf.one_hot(tf.stack(state[2], axis=0), 6, dtype="float16")
            actionPStack = state[3]

            cLoss = self.update_agent_weights(
                obvStack, rewardStack, actionStack, actionPStack, 1.)
            print(f"loss:{cLoss}")

            del obvStack
            del rewardStack
            del actionStack
            del actionPStack

        del obvStacks
        del rewardStacks
        del actionStacks
        del actionPStacks
        del stateDataset

        self.replayBuffer = self.replayBuffer.copy()

        # # training more on hight score recorders
        # if(len(self.hScoreReplayBuffer) > 0):
        #     obvStacks, rewardStacks, actionStacks, actionPStacks = zip(
        #         *(self.hScoreReplayBuffer))

        #     with tf.device('/GPU:0'):
        #         stateDataset = tf.data.Dataset.from_tensor_slices(
        #             (list(obvStacks), list(rewardStacks), list(actionStacks), list(actionPStacks)))
        #         stateDataset = stateDataset.batch(
        #             self.bs, drop_remainder=True).repeat(2).shuffle(32000)

        #     for state in stateDataset:
        #         # policy gradient training
        #         obvStack = state[0]
        #         rewardStack = state[1]
        #         actionStack = tf.one_hot(tf.stack(state[2], axis=0), 6, dtype="float16")
        #         actionPStack = state[3]

        #         cLoss = self.update_agent_weights(
        #             obvStack, rewardStack, actionStack, actionPStack, 1.)
        #         print(f"loss:{cLoss}")
        #     del obvStacks
        #     del rewardStacks
        #     del actionStacks
        #     del actionPStacks
        #     del stateDataset
        #     self.hScoreReplayBuffer = self.hScoreReplayBuffer.copy()

    @tf.function(jit_compile=True)
    def update_agent_weights(self, obvStack, rewardStack, actionStack, actionPStack, rewardWeight):
        with tf.device('/GPU:0'):
            with tf.GradientTape() as grad:
                predicts = self.agent(obvStack)

                ## entropy regularization
                er = k.ops.mean(k.ops.sum(predicts * tf.math.log(predicts), axis=-1))

                # # croping the agent position in the image => this doesn't make sense
                # agentPos = k.layers.Cropping2D(cropping=((180, 15), (0, 0)))(obvStack)
                # agentPosDiff = k.ops.mean(k.ops.absolute(agentPos[0:int(self.bs/2)] - agentPos[int(self.bs/2):]))

                # # watching the action (on-policy)
                # maskedPredictsLROnPolicy = k.ops.sum(k.ops.cast(tf.one_hot(k.ops.argmax(predicts), 6), k.mixed_precision.dtype_policy().variable_dtype) *
                #                              tf.constant([0,0,1,1,0,0], dtype=k.mixed_precision.dtype_policy().variable_dtype), axis=-1) * rewardWeight
                maskedPredictsLROnPolicy = tf.cast(0., k.mixed_precision.dtype_policy().variable_dtype)

                # counting the action ce between batch (on-policy, MC)
                # actionCE = tf.clip_by_value(
                #     -k.ops.mean(
                #     k.ops.cast(tf.one_hot(k.ops.argmax(predicts[0:int(self.bs/2)]), 6), k.mixed_precision.dtype_policy().variable_dtype) *
                #     k.ops.log(k.ops.clip(predicts[int(self.bs/2):], 1e-6, 1)) +
                #     k.ops.cast(tf.one_hot(k.ops.argmax(predicts[int(self.bs/2):]), 6), k.mixed_precision.dtype_policy().variable_dtype) *
                #     k.ops.log(k.ops.clip(predicts[0:int(self.bs/2)], 1e-6, 1))
                #     ),
                #     0,
                #     100)
                # actionCE += -k.ops.mean(
                #     k.ops.cast(predicts[0:int(self.bs/2)], k.mixed_precision.dtype_policy().variable_dtype) *
                #     k.ops.log(k.ops.clip(predicts[int(self.bs/2):], 1e-6, 1)) +
                #     k.ops.cast(predicts[int(self.bs/2):], k.mixed_precision.dtype_policy().variable_dtype) *
                #     k.ops.log(k.ops.clip(predicts[0:int(self.bs/2)], 1e-6, 1))
                #     )
                actionCE = tf.cast(0., k.mixed_precision.dtype_policy().variable_dtype)

                # counting the action variance between batch (on-policy, MC)
                actionVari = tf.math.reduce_mean(tf.math.reduce_variance(predicts, axis=0))
                # actionVari = tf.cast(0., k.mixed_precision.dtype_policy().variable_dtype)

                on_policy_action_ce = tf.reduce_sum(
                    k.ops.cast(tf.one_hot(k.ops.argmax(predicts), 6), k.mixed_precision.dtype_policy().variable_dtype) *
                    -tf.math.log(tf.clip_by_value(predicts, 1e-6, 1.)), axis=-1)
                on_policy_ce = tf.reduce_mean(tf.stop_gradient(actionCE) * on_policy_action_ce + tf.stop_gradient(maskedPredictsLROnPolicy + actionVari) * on_policy_action_ce)

                # importance sampling (off-policy)
                under = actionPStack
                upper = tf.math.reduce_max(
                    predicts * actionStack, axis=-1)
                iSampling = tf.cast(upper/under, dtype='float16')
                clippedReward = tf.clip_by_value(tf.cast(tf.stop_gradient(rewardStack), dtype='float16') * tf.stop_gradient(iSampling), -100, 500)
                clippedReward *= rewardWeight

                # # watching the action (off-policy)
                # maskedPredictsLROffPolicy = k.ops.sum(
                #     (actionStack * tf.constant([0,0,1,1,0,0], dtype=k.mixed_precision.dtype_policy().variable_dtype)), axis=-1) * tf.stop_gradient(iSampling)
                maskedPredictsLROffPolicy = 0.

                off_policy_action_ce = tf.reduce_sum(
                    actionStack * -tf.math.log(tf.clip_by_value(predicts, 1e-6, 1.)), axis=-1)
                off_policy_ce = tf.reduce_mean(clippedReward * off_policy_action_ce + maskedPredictsLROffPolicy * off_policy_action_ce)

                total_loss = on_policy_ce + off_policy_ce + er * 1e-1

                gradients = grad.gradient(
                    total_loss + tf.reduce_sum(self.agent.losses) , self.agent.trainable_variables)
                self.optimizer.apply(
                    gradients, self.agent.trainable_variables)

        return total_loss

    def infinity_training(self):
        while(1):
            self.pooling_sampling()
            self.agent_learning()
            self.agent.save("si_agent.keras")

    def greedy_until_training(self, greedy_upper=.2, greedy_lower=.02):
        self.greedy = greedy_upper
        while(self.greedy > greedy_lower):
            self.pooling_sampling()
            self.agent_learning()
            self.agent.save("si_agent.keras")



def main():
    # k.mixed_precision.set_global_policy('mixed_float16')
    k.mixed_precision.set_global_policy('float16')

    # ag = agent()
    ag = k.saving.load_model("si_agent.keras")

    # env = atari_trainer(ag, epiNo=15, cloneAgFunc=agent)
    # # env.infinity_training()
    # env.greedy_until_training(greedy_upper=.2, greedy_lower=.02)

    env = atari_trainer(ag, epiNo=15, cloneAgFunc=agent)
    for loop_conter in range(20):
        env.pooling_sampling()
        env.agent_learning()
        ag.save("si_agent.keras")


if __name__ == "__main__":
    main()
