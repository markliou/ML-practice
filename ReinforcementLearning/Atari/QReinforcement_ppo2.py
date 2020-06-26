import gym 
import tensorflow as tf
import numpy as np
import os
import random
from collections import deque

try:
    tf = tf.compat.v1
    tf.disable_eager_execution()
except ImportError:
    pass

def conv2d(X, channel_no = 64, kernel_size = 3, stride_no = 1):
    return tf.layers.conv2d(X, channel_no, 
                               [kernel_size, kernel_size], 
                               [stride_no, stride_no], 
                               padding='SAME', 
                               kernel_initializer=tf.keras.initializers.glorot_normal,
                               activation=tf.nn.relu
                               )
pass

def Q(S):
    S = (tf.cast(S,tf.float32)-128)/128.  #(210, 160, 3)
    #S = tf.layers.Dropout(.5)(S)
    S = tf.layers.conv2d(S,16, [1,1], [1,1], padding='SAME', activation=tf.nn.relu)
    conv1 = conv2d(S, stride_no=2) #(105, 80)
    #conv1 = tf.layers.Dropout(.5)(conv1)
    conv2 = conv2d(conv1, stride_no=2) #(53, 40)
    #conv2 = tf.layers.Dropout(.5)(conv2)
    conv3 = conv2d(conv2, stride_no=2) #(27, 20)
    #conv3 = tf.layers.Dropout(.5)(conv3)
    conv4 = conv2d(conv3, 128, stride_no=2) #(14, 10)
    #conv4 = tf.layers.Dropout(.5)(conv4)
    conv5 = conv2d(conv4, 256, stride_no=2) #(7, 5)
    #conv5 = tf.layers.Dropout(.5)(conv5)
    conv6 = conv2d(conv5, 512, stride_no=2) #(4, 3)
    #conv6 = tf.layers.Dropout(.5)(conv6)
    
    f1 = tf.layers.flatten(conv6)
    #f1 = tf.layers.Dropout(.5)(f1)
    f2 = tf.layers.dense(f1, 1024, activation=tf.nn.relu)
    #f2 = tf.layers.Dropout(.5)(f2)
    f3 = tf.layers.dense(f2, 512, activation=tf.nn.relu)
    out = tf.layers.dense(f3, 4)

    return out
pass

# Enviroment settings
STEP_LIMIT = 1000
EPISODE = 1000
EPSILONE = .8
REWARD_b = .0
REWARD_NORMA = 500 # because the peak reward is close to 500, empiritically
GAMMA = .5
ALPHA = .9
PPO_EPSILON = .5
DIE_PANELTY = 0
WARMING_EPI = 0
BEST_REC = 0.
BEST_STEPS = 1.
STATE_GAMMA = .8
REPLAY_BUFFER = []
Loss = 0
SCORE_REC_FILE = 'score_rec2.txt'

env = gym.make('SpaceInvaders-v0') 
os.system("echo > {}".format(SCORE_REC_FILE)) #clean the previoud recorders

# Actor settings
Opt_size = 16 # skip frames
OPT_FLAG = False
Act_S = tf.placeholder(tf.int8, [None, 210, 160, 3])
Act_Sp = tf.placeholder(tf.int8, [None, 210, 160, 3])
Act_R = tf.placeholder(tf.float32, [None])
Act_pi = tf.placeholder(tf.float32, [None])
Actions4Act = tf.placeholder(tf.uint8, [None])
Actions4Act_oh = tf.one_hot(Actions4Act, 4) 

Act_A = Q(Act_S)
pi = tf.reduce_max(tf.nn.softmax(Act_A) * Actions4Act_oh) 
Command_A = tf.argmax(Act_A, axis=-1)

Act_Ap = Q(Act_Sp)
pip = tf.reduce_max(tf.nn.softmax(Act_Ap) * Actions4Act_oh) 

rho = tf.clip_by_value((pip/Act_pi),1 - PPO_EPSILON , 1 + PPO_EPSILON)
#rho = tf.log(pip + Act_pi) - tf.log(pip)

#PPO_R = tf.reduce_min(tf.concat([tf.expand_dims(Act_R, axis=-1), tf.expand_dims(Act_R * rho, axis=-1)], axis=1), axis=-1)
PPO_PL = tf.reduce_mean(tf.pow((Act_R + tf.reduce_max(Act_A) - tf.reduce_max(Act_Ap * Actions4Act_oh)) * rho, 2))
PL = tf.reduce_mean(tf.pow((Act_R + tf.reduce_max(Act_A) - tf.reduce_max(Act_Ap * Actions4Act_oh)), 2)) #Q 

#Opt = tf.train.RMSPropOptimizer(1E-4, momentum=.0, centered=True).minimize(PL)
#Opt = tf.train.MomentumOptimizer(learning_rate=1E-6, momentum=.8).minimize(PL)

optimizer = tf.train.RMSPropOptimizer(1E-4, momentum=.6, centered=True)
#gr, va = zip(*optimizer.compute_gradients(PL))
gr, va = zip(*optimizer.compute_gradients(PPO_PL))
gr = [None if gr is None else tf.clip_by_norm(grad, 10.) for grad in gr]
Opt = optimizer.apply_gradients(zip(gr, va))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

episode = 0
while(1):
    episode += 1
    Rp = 0.
    S = env.reset() #(210, 160, 3)
    GameScore = 0
    Clives = 3
    Reward_cnt = 0.
    CuReward = 0.
    R_list, S_list = [],[]
    Shooting_S = []
    
    steps = 0
    if (np.random.random() >= EPSILONE/np.clip(episode-WARMING_EPI,1E-9,None)) or (WARMING_EPI < episode) :
        Greedy_flag = False 
    else:
        Greedy_flag = True 
    pass
    while(1):
        steps += 1
    # for step in range(STEP_LIMIT):
        #env.render() # show the windows. If you don't need to monitor the state, just comment this.
        # print(S)
        
        # A = env.action_space.sample() # random sampling the actions
        # print(A)

        # sampling action from Q
        # epsilon greedy
        # actions: [noop, fire, right, left, right fire, left fire] 
        if Greedy_flag or (np.random.random() < .2):
            A = np.random.randint(4) # exlude right fire and left fire, such combo actions
        else:
            A = sess.run(Command_A, feed_dict={Act_S:np.array(S).reshape([1, 210, 160, 3])})[0]
        pass
        # print(A) # monitor the action

        # get pi
        pi_c = sess.run(pi, feed_dict={Act_S:np.array(S).reshape([1, 210, 160, 3]),
                                       Actions4Act:np.array([A])
                                      }
                        )
        #print(pi_c) # monitor the pi

        Sp = S.copy()
        S, R, finish_flag, info = env.step(A)
        GameScore += R
        S = np.clip(Sp * STATE_GAMMA *.5 + S * .5, 0, 255)  # keep the previoud state as input would be creating a RNN like condition
       
        # handling the reward and actions
        if R > 0:
            OPT_FLAG = True
        pass
        
        #if A in [0]:
        #    R += REWARD_b * .5 # give the reward for moving. This would be helpful for telling agent to avopod bullet
        #elif A in [2,3]:
        #    R += REWARD_b * .1
        #elif A in [1]:
        #    Shooting_S.append([Sp, A, S]) # s, s'. Treat it as MC and memory replay hybrid
        pass

        Shooting_S.append([Sp, A, S, pi_c]) # s, s'. Treat it as MC and memory replay hybrid

        # advantage
        #Reward_cnt = GAMMA * pow((R - Rp),2)
        #Reward_cnt = GAMMA * R - Rp   
        #Reward_cnt = GAMMA * np.clip(R - Rp, 0, np.inf)
        #print(Reward_cnt)

        #Rp = Reward_cnt 
        #print(R)

        # CuReward = CuReward * GAMMA + R
        # CuReward = ALPHA * CuReward + R
        CuReward = R 
        # CuReward = 1 + Reward_cnt # normalized reward with game score
        # CuReward += Reward_cnt
        # CuReward = CuReward * GAMMA + Reward_cnt
        # CuReward = CuReward * GAMMA + (Reward_cnt - (BEST_REC/BEST_STEPS))
        #print(CuReward)

        # print('Reward:{}'.format(R)) # the reward will give this action will get how much scores. it's descreted.
        # print('Info:{}'.format(info['ale.lives'])) # info in space invader will give the lives of the current state

        if finish_flag or (Clives > info['ale.lives']):
            Clives = info['ale.lives']
            # CuReward = ALPHA * CuReward - DIE_PANELTY 
            # CuReward = 0
            # CuReward = np.clip(CuReward, 0, None)
            # print('This episode is finished ...')
            A = sess.run(Command_A, feed_dict={Act_S:np.array(Sp).reshape([1, 210, 160, 3])})[0]
            #Loss, _ = sess.run([PL, Opt], 
            #                   feed_dict={
            #                              Act_S:np.array(S).reshape([-1, 210, 160, 3]),
            #                              Act_Sp:np.array(Sp).reshape([-1, 210, 160, 3]),
            #                              Act_R:np.array(CuReward).reshape([-1]),
            #                              Actions4Act:np.array(A).reshape([-1]) 
            #                             }
            #                   )
            
            #if len(Shooting_S) > 0: # ponishing the shutting state because of die
            #    for Si, Ai, Spi in Shooting_S:
            #        Loss, _ = sess.run([PL, Opt],
            #                feed_dict={
            #                    Act_S:np.array(Si).reshape([-1, 210, 160, 3]),
            #                    Act_Sp:np.array(Spi).reshape([-1, 210, 160, 3]),
            #                    Act_R:np.array((0/len(Shooting_S)) - REWARD_b * .0).reshape([-1]),
            #                    Actions4Act:np.array(Ai).reshape([-1])
            #                    }
            #                )
            #    pass
            #    Shooting_S = []
            #pass
            
            if finish_flag:
                if BEST_REC < GameScore:
                    BEST_REC = GameScore 
                pass
                if BEST_STEPS < steps:
                    BEST_STEPS = steps 
                pass
                #if REWARD_b < (GameScore/steps) :
                #    #REWARD_b = (GameScore/steps)
                #    REWARD_b = (CuReward/steps)
                #pass
                #if REWARD_b < (CuReward/steps):
                #    REWARD_b = CuReward/steps
                #pass
                if DIE_PANELTY < CuReward:
                    DIE_PANELTY = CuReward * .8 
                    #print(DIE_PANELTY)
                pass
                os.system("echo {} >> {}".format(GameScore, SCORE_REC_FILE))
                break
            else:
                continue
            pass
            #Shooting_S = []
            #Sp = np.zoeo([210, 160, 3])
        pass 
        # TD
        #Loss = np.nan
        if  (OPT_FLAG and len(Shooting_S) > 0): # shooting MC
            SN = len(Shooting_S)
            #print('SN {}'.format(SN))
            #SR = (R/SN) - REWARD_b * .8
            tf.clip_by_value(CuReward, 0, 50)
            SR = CuReward/SN
            #SR = (R)/SN 
            #print('SR {}'.format(SR))
            CURRENT_BUFFER = []

            if REWARD_b < (GameScore/steps) :
                REWARD_b = (CuReward/len(Shooting_S))
            pass

            for Si, Ai, Spi, pii in (Shooting_S):
                # push information into replay buffer
                if (np.random.random() > .2) and (SR > REWARD_b * .2) :
                    CURRENT_BUFFER.append([Spi, Ai, Si, SR, pii])
                    if len(REPLAY_BUFFER) < 1.5E4:
                        REPLAY_BUFFER.append([Spi, Ai, Si, SR, pii])
                    #elif (SR > REWARD_b * .8):
                    else:
                        #REPLAY_BUFFER[np.random.randint(len(REPLAY_BUFFER))] = [Spi, Ai, Si, SR, pii]
                        REPLAY_BUFFER.reverse()
                        REPLAY_BUFFER.pop()
                        REPLAY_BUFFER.reverse()
                        REPLAY_BUFFER.append([Spi, Ai, Si, SR, pii])
                    pass
                pass
                OPT_FLAG = False
                Shooting_S = []
            pass

            #random.shuffle(REPLAY_BUFFER)
            #if (GameScore * .5 > BEST_REC) or (steps * .5 > BEST_STEPS): 
            if True:
                training_buffer = CURRENT_BUFFER + REPLAY_BUFFER[:int(len(REPLAY_BUFFER) * .2)]
            else:
                training_buffer = REPLAY_BUFFER
            pass
            random.shuffle(training_buffer)
            Sib, Aib, Spib, Rib, piib = [], [], [], [], []
            for Spi, Ai, Si, SR, pii in (training_buffer):
                Sib.append(Si)
                Aib.append(Ai)
                Spib.append(Spi)
                Rib.append(SR - REWARD_b * .1)
                piib.append(pii)
                if len(Aib) == 32:
                    Loss, _ = sess.run([PL, Opt], 
                                    feed_dict={
                                            Act_S:np.array(Sib).reshape([-1, 210, 160, 3]),
                                            Act_Sp:np.array(Spib).reshape([-1, 210, 160, 3]),
                                            Act_R:np.array(Rib).reshape([-1]),
                                            Act_pi:np.array(piib).reshape([-1]),
                                            Actions4Act:np.array(Aib).reshape([-1])
                                            }
                                   )
                    Sib, Aib, Spib, Rib , piib = [], [], [], [], []
                pass
            pass
        pass

        #print('Action:{}  Loss:{} Epsilon:{} greedy:{} score:{}'.format(A, Loss, EPSILONE/np.clip(episode-WARMING_EPI,1E-9,None), Greedy_flag, GameScore))
    pass

    # memory replay
    #random.shuffle(REPLAY_BUFFER)
    #for m in REPLAY_BUFFER:
    #    Spm, Am, Sm, SRm = m
    #    _ = sess.run([PL, Opt],
    #            feed_dict={
    #                Act_S:np.array(Sm).reshape([-1, 210, 160, 3]),
    #                Act_Sp:np.array(Spm).reshape([-1, 210, 160, 3]),
    #                Act_R:np.array(SRm).reshape([-1]),
    #                Actions4Act:np.array(Am).reshape([-1])
    #                      }
    #            )
    #pass

    # random keep memory
    #if (len(REPLAY_BUFFER) > 1E5):
    #    REPLAY_BUFFER = REPLAY_BUFFER[0:int(np.random.random() * len(REPLAY_BUFFER))]
    #pass

    print("Epi:{}  Score:{}  Loss:{}  Reward:{}  steps:{}  memory:{}".format(episode, GameScore, Loss, CuReward, steps, len(REPLAY_BUFFER)))

pass
