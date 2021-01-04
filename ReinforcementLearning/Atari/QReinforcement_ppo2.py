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

def conv2d(X, name, channel_no = 64, kernel_size = 3, stride_no = 1, activation=tf.nn.relu):
    #with tf.variable_scope('conv_b', reuse=False):    
     return tf.layers.conv2d(X, channel_no, 
                                   [kernel_size, kernel_size], 
                                   [stride_no, stride_no], 
                                   padding='SAME', 
                                   kernel_initializer=tf.keras.initializers.glorot_normal,
                                   activation=activation,
                                   reuse=False,
                                   name=name
                                   )
pass

def Q(S):
    with tf.variable_scope('Q', reuse=tf.AUTO_REUSE): 
    #if True:
        So = (tf.cast(S, tf.float32)-128)/128.  #(210, 160, 3)
        #Sr = tf.image.resize(So, (110, 84))[13:110 - 13, :]
        #S = tf.layers.Dropout(.5)(S)
        Sf = tf.layers.conv2d(So, 16, [1,1], [1,1], padding='SAME', activation=tf.nn.relu, reuse=False, name='So')
        conv1 = conv2d(So, stride_no=2, name='conv1') #(105, 80)
        #conv1 = tf.layers.Dropout(.5)(conv1)
        conv2 = conv2d(conv1, stride_no=2, name='conv2') #(53, 40)
        #conv2 = tf.layers.Dropout(.5)(conv2)
        conv3 = conv2d(conv2, stride_no=2, name='conv3') #(27, 20)
        #conv3 = tf.layers.Dropout(.5)(conv3)
        conv4 = conv2d(conv3, channel_no=128, stride_no=2, name='conv4') #(14, 10)
        #conv4 = tf.layers.Dropout(.5)(conv4)
        conv5 = conv2d(conv4, channel_no=256, stride_no=2, name='conv5') #(7, 5)
        #conv5 = tf.layers.Dropout(.5)(conv5)
        conv6 = conv2d(conv5, channel_no=512, stride_no=2, activation=None, name='conv6') #(4, 3)
        #conv6 = tf.layers.Dropout(.5)(conv6)

        conv6g = conv2d(conv5, channel_no=512, stride_no=2, activation=tf.nn.softmax, name='conv6g')
        conv6 *= conv6g 
    
        f1 = tf.layers.flatten(conv6)
        #f1 = tf.layers.Dropout(.5)(f1)
        f2 = tf.layers.dense(f1, 256, activation=tf.nn.relu)
        #f2 = tf.layers.Dropout(.5)(f2)
        for i in range(5):
            f2 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(f2)
            #f2 = tf.layers.Dropout(.5)(f2)
            #f2 = tf.keras.layers.LayerNormalization()(f2)
        pass
        f3 = tf.layers.dense(f2, 512, activation=tf.nn.relu)
        out = tf.layers.dense(f3, 4)

    return out
pass

def tQ(S):
    with tf.variable_scope('tQ', reuse=tf.AUTO_REUSE):
    #if True:
        So = (tf.cast(S, tf.float32)-128)/128.  #(210, 160, 3)
        #Sr = tf.image.resize(So, (110, 84))[13:110 - 13, :]
        #S = tf.layers.Dropout(.5)(S)
        Sf = tf.layers.conv2d(So, 16, [1,1], [1,1], padding='SAME', activation=tf.nn.relu, reuse=False, name='So')
        conv1 = conv2d(So, stride_no=2, name='conv1') #(105, 80)
        conv1 = tf.layers.Dropout(.5)(conv1)
        conv2 = conv2d(conv1, stride_no=2, name='conv2') #(53, 40)
        conv2 = tf.layers.Dropout(.5)(conv2)
        conv3 = conv2d(conv2, stride_no=2, name='conv3') #(27, 20)
        conv3 = tf.layers.Dropout(.5)(conv3)
        conv4 = conv2d(conv3, channel_no=128, stride_no=2, name='conv4') #(14, 10)
        conv4 = tf.layers.Dropout(.5)(conv4)
        conv5 = conv2d(conv4, channel_no=256, stride_no=2, name='conv5') #(7, 5)
        conv5 = tf.layers.Dropout(.5)(conv5)
        conv6 = conv2d(conv5, channel_no=512, stride_no=2, activation=None, name='conv6') #(4, 3)
        #conv6 = tf.layers.Dropout(.5)(conv6)

        conv6g = conv2d(conv5, channel_no=512, stride_no=2, activation=tf.nn.softmax, name='conv6g')
        conv6 *= conv6g

        f1 = tf.layers.flatten(conv6)
        #f1 = tf.layers.Dropout(.5)(f1)
        f2 = tf.layers.dense(f1, 256, activation=tf.nn.relu)
        #f2 = tf.layers.Dropout(.5)(f2)
        for i in range(5):
            f2 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(f2) 
            f2 = tf.layers.Dropout(.2)(f2)
            #f2 = tf.keras.layers.LayerNormalization()(f2)
        pass
        f3 = tf.layers.dense(f2, 512, activation=tf.nn.relu)
        out = tf.layers.dense(f3, 4)

    return out
pass


STEP_LIMIT = 1000
EPISODE = 1000
EPSILONE = .8
REWARD_b = .0
REWARD_NORMA = 500 # because the peak reward is close to 500, empiritically
GAMMA = .5
ALPHA = .99
PPO_EPSILON = .2
DIE_PANELTY = 0
WARMING_EPI = 0
BEST_REC = 0.
BEST_STEPS = 1.
STATE_GAMMA = 1.
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
Act_At = tQ(Act_S)
pi = tf.reduce_max(tf.nn.softmax(Act_A) * Actions4Act_oh) 
#Command_A = tf.argmax(Act_A, axis=-1)
Command_A = tf.argmax(Act_At, axis=-1)

Act_Ap = Q(Act_Sp)
Act_Apt = tQ(Act_Sp)
#Act_Apt = tQ(Act_Sp + tf.random.uniform(tf.shape(Act_Sp), .0, .1))
pip  = tf.reduce_max(tf.nn.softmax(Act_Ap) * Actions4Act_oh) 
pipt = tf.reduce_max(tf.nn.softmax(Act_Apt) * Actions4Act_oh)

Q_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q')
tQ_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tQ')
update_Q = [tf.assign(q, tq) for q,tq in zip(Q_weights,tQ_weights)]
update_tQ = [tf.assign(tq, (q * 1e-5 + tq * (1-1e-5))) for q,tq in zip(Q_weights,tQ_weights)]
#update_tQ = [tf.assign(tq, q) for q,tq in zip(Q_weights,tQ_weights)]

#rho = tf.clip_by_value((pip/Act_pi),1 - PPO_EPSILON , 1 + PPO_EPSILON)
#rho = tf.clip_by_value((pipt/pip),1 - PPO_EPSILON , 1 + PPO_EPSILON)
#rho = tf.clip_by_value((pip/pipt),0 , 1 + PPO_EPSILON)
#rho = tf.log(pip + Act_pi) - tf.log(pip)
rho = 10

#PPO_R = tf.reduce_min(tf.concat([tf.expand_dims(Act_R, axis=-1), tf.expand_dims(Act_R * rho, axis=-1)], axis=1), axis=-1)
#PPO_PL = tf.reduce_mean(tf.pow((Act_R + tf.reduce_max(Act_A) - tf.reduce_max(Act_Ap * Actions4Act_oh)), 2) * rho) 
#PPO_PL = tf.reduce_mean(tf.pow((Act_R + tf.reduce_max(Act_A) - tf.reduce_max(Act_Ap * Actions4Act_oh)), 2) - rho)
#PPO_PL = tf.clip_by_value(tf.reduce_mean(tf.pow((Act_R + tf.reduce_max(Act_A) - tf.reduce_max(Act_Ap * Actions4Act_oh)), 2)), rho, -rho)
#PPO_PL = tf.reduce_mean(tf.clip_by_value(tf.pow((Act_R + tf.reduce_max(Act_A) - tf.reduce_max(Act_Ap * Actions4Act_oh)), 2), rho, -rho))
PPO_PL = tf.reduce_mean(tf.clip_by_value(tf.pow((Act_R + tf.reduce_max(Act_A) - tf.reduce_max(Act_Ap * Actions4Act_oh)), 2), -rho, rho))
PL = tf.reduce_mean(tf.pow((Act_R + tf.reduce_max(Act_A) - tf.reduce_max(Act_Ap * Actions4Act_oh)), 2)) #Q

#Opt = tf.train.RMSPropOptimizer(1E-4, momentum=.0, centered=True).minimize(PPO_PL)
#Opt = tf.train.MomentumOptimizer(learning_rate=1E-6, momentum=.8).minimize(PL)
#Opt = tf.train.RMSPropOptimizer(1E-4, momentum=.9, centered=False).minimize(PPO_PL, var_list=tQ_weights)

#optimizer = tf.train.MomentumOptimizer(1E-3, momentum=.0)
optimizer = tf.train.RMSPropOptimizer(1E-4, momentum=.6, decay=.9, centered=True)
#gr, va = zip(*optimizer.compute_gradients(PL))
gr_va = optimizer.compute_gradients(PPO_PL, var_list=Q_weights)
#gr_va = optimizer.compute_gradients(PL, var_list=Q_weights)
capped_gvs = [(grad if grad is None else tf.clip_by_norm(grad, clip_norm=10.), var) for grad, var in gr_va]
#gr = [None if gr is None else tf.clip_by_norm(grad, 1.) for grad in gr]
#gr = [grad if gr is None else tf.clip_by_norm(grad, .5) for grad in gr]
Opt = optimizer.apply_gradients(capped_gvs)

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
    #CuReward = -REWARD_b * .2
    R_list, S_list = [],[]
    Shooting_S = []
    
    steps = 0
    if (np.random.random() >= EPSILONE/np.clip(episode-WARMING_EPI,1E-9,None)) or (WARMING_EPI < episode) :
        Greedy_flag = False 
    else:
        Greedy_flag = True 
    pass

    exploration = .03
    while(1):
        steps += 1
    # for step in range(STEP_LIMIT):
        # env.render() # show the windows. If you don't need to monitor the state, just comment this.
        # print(S)
        
        # A = env.action_space.sample() # random sampling the actions
        # print(A)

        # sampling action from Q
        # epsilon greedy
        # actions: [noop, fire, right, left, right fire, left fire] 
        if exploration > 1E-5:
            exploration *= .999
        pass
        #if (np.random.random() < .05):
        if (np.random.random() < exploration):
        # if Greedy_flag or (np.random.random() < .2):
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
        #S = Sp
       
        # handling the reward and actions
        if R > 0:
            OPT_FLAG = True
        pass
        #OPT_FLAG = True
        
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
        CuReward = ALPHA * CuReward + R
        CuReward += .05 #step reward 
        # CuReward = R 
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
            CuReward = 0
            # CuReward = np.clip(CuReward, 0, None)
            # print('This episode is finished ...')
            A = sess.run(Command_A, feed_dict={Act_S:np.array(Sp).reshape([1, 210, 160, 3])})[0]
            
            OPT_FLAG = True

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
                sess.run(update_tQ)
                #CuReward -= REWARD_b
                if BEST_REC < GameScore:
                    BEST_REC = GameScore 
                   # sess.run(update_tQ)
                #else :
                    #sess.run(update_Q)
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
                    DIE_PANELTY = CuReward * .9 
                    #print(DIE_PANELTY)
                pass
                os.system("echo {} >> {}".format(GameScore, SCORE_REC_FILE))
                break
            else:
                continue
            pass
            #Shooting_S = []
            Sp = np.zoeo([210, 160, 3])
            #CuReward -= REWARD_b * .2
        pass 
        # TD
        #Loss = np.nan
        #if (OPT_FLAG and len(Shooting_S) > 0): # shooting MC
        if len(Shooting_S) > 0:
            SN = len(Shooting_S)
            #print('SN {}'.format(SN))
            #SR = (R/SN) - REWARD_b * .8
            #tf.clip_by_value(CuReward, 0, 50) # reward clipping
            SR = CuReward/SN
            #SR = (R)/SN 
            #print('SR {}'.format(SR))
            CURRENT_BUFFER = []

            #REWARD_b *= 0.99
            #if REWARD_b < CuReward/len(Shooting_S): #(CuReward/steps): #(GameScore/steps) :
            #if (REWARD_b < CuReward) and OPT_FLAG:
            if (REWARD_b < CuReward):
            #if 1:
                #REWARD_b = (CuReward/len(Shooting_S))
                #REWARD_b = CuReward/steps
                #REWARD_b = (CuReward/len(Shooting_S)) * .1 + REWARD_b * .9
                REWARD_b = CuReward
                REPLAY_BUFFER = []
            pass
        pass 
        if (OPT_FLAG):
            for Si, Ai, Spi, pii in (Shooting_S):
                # push information into replay buffer
                if (np.random.random() > .2) :#and (SR > REWARD_b * .2) :
                    CURRENT_BUFFER.append([Spi, Ai, Si, SR, pii])
                    if len(REPLAY_BUFFER) < 1E4:
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
            training_buffer = REPLAY_BUFFER.copy()
            random.shuffle(training_buffer)
            training_buffer = CURRENT_BUFFER + training_buffer[:int(len(training_buffer) * .6)]
           
            #for reward normalization
            if len(training_buffer) > 1:
                reward_np_array = np.array(training_buffer)[:,3]
                reward_mean = reward_np_array.mean()
                reward_std  = reward_np_array.std()
                reward_hq = np.quantile(reward_np_array, .8)
                reward_lq = np.quantile(reward_np_array, .2) 
            else:
                reward_mean = reward_hq = reward_lq = 0
                reward_std = 1
            pass

            Sib, Aib, Spib, Rib, piib = [], [], [], [], []
            for training_loop_cnt in range(5):    
                random.shuffle(training_buffer)
                for Spi, Ai, Si, SR, pii in (training_buffer):
                    Sib.append(Si)
                    Aib.append(Ai)
                    Spib.append(Spi)
                    #Rib.append(SR - REWARD_b * .2)
                    #Rib.append(SR)
                    #Rib.append((SR - reward_mean * 1.5)/reward_std)
                    Rib.append((SR - reward_lq)/(reward_hq - reward_lq + 0.00000001) * 2 - .2)
                    #Rib.append(SR - reward_hq)
                    piib.append(pii)
                    if len(Aib) == 128:
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
            #sess.run(update_tQ)
            pass
            #sess.run(update_tQ)
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
        
    #sess.run(update_tQ)
    #sess.run(update_Q)
pass
