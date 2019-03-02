# this version try to use the previous actions as condition
#   markliou 2019/2/26

import gym 
import tensorflow as tf
import numpy as np

def conv2d(X, name, kernel_size = 3, stride_no = 1, reuse = False, trainable = True):
    return tf.layers.conv2d(X, 32, 
                               [kernel_size, kernel_size], 
                               [stride_no, stride_no], 
                               padding='SAME', 
                               kernel_initializer=tf.keras.initializers.glorot_normal,
                               activation=tf.nn.elu,
                               name = name,
                               trainable = trainable,
                               reuse=reuse
                               )
pass

def Q(S, AH, SH, reuse=False, trainable = True):
    
    S = (tf.cast(S,tf.float32)-128)/128.  #(210, 160, 3)
    SH = tf.stack([tf.concat(tf.unstack(SH, axis=0), axis=-1)], axis=0)
    SH = (tf.cast(SH,tf.float32)-128)/128.
    AH = tf.reshape(AH, [-1, AH.shape[1] * AH.shape[2]]) #[None, 32, 6]
    AH = AH * 2 - 1

    conv1 = conv2d(tf.concat([S, SH], axis=-1), stride_no=2, reuse=reuse, trainable = trainable, name='conv1') #(105, 80)
    conv2 = conv2d(conv1, stride_no=2, reuse=reuse, trainable = trainable, name='conv2') #(53, 40)
    conv3 = conv2d(conv2, stride_no=2, reuse=reuse, trainable = trainable, name='conv3') #(27, 20)
    conv4 = conv2d(conv3, stride_no=2, reuse=reuse, trainable = trainable ,name='conv4') #(14, 10)
    conv5 = conv2d(conv4, stride_no=2, reuse=reuse, trainable = trainable, name='conv5') #(7, 5)
    conv6 = conv2d(conv5, stride_no=2, reuse=reuse, trainable = trainable, name='conv6') #(4, 3)
    
    f1 = tf.layers.flatten(conv6)
    f2 = tf.layers.dense(tf.concat([f1, AH], axis=-1), 64, activation=tf.nn.elu, trainable = trainable, reuse=reuse, name='f2')
    f3 = tf.layers.dense(f2, 32, activation=tf.nn.elu, trainable = trainable, reuse=reuse, name='f3')
    out = tf.layers.dense(f3, 6, trainable = trainable, reuse=reuse, name='out')

    return out
pass

def KL_div_with_normal(X):
    ## X with the shape of (n, l)
    X = np.sum(X, axis=0)/X.shape[0]
    return np.clip(np.sum(X * (np.log(X+1E-9) - np.log(1/X.shape[0]))), 0, None)
pass

# Enviroment settings
STEP_LIMIT = 1000
EPISODE = 1000
EPSILONE = 1.
REWARD_b = .1
REWARD_NORMA = 50 # because the peak reward is close to 500, empiritically
GAMMA = .95
DIE_PANELTY = .2
WARMING_EPI = 0

env = gym.make('SpaceInvaders-v0') 

# Actor settings
action_memo = 64
Act_S = tf.placeholder(tf.int8, [None, 210, 160, 3])
Act_R = tf.placeholder(tf.float32, [None])
Act_m = tf.placeholder(tf.float32, [None, action_memo, 6])
Sta_m = tf.placeholder(tf.int8, [action_memo, 210, 160, 3])
Actions4Act = tf.placeholder(tf.uint8, [None])
Actions4Act_oh = tf.one_hot(Actions4Act, 6) 

Act_A = Q(Act_S, Act_m, Sta_m, trainable = True, reuse=False)
Act_sample = tf.argmax(tf.nn.softmax(Act_A), axis=-1)[0]

# PL = Act_R * -tf.log(tf.reduce_sum(tf.nn.softmax(Act_A) * Actions4Act_oh)+1E-9)
PL = (Act_R/REWARD_NORMA) * tf.nn.softmax_cross_entropy_with_logits_v2(labels=Actions4Act_oh, logits=Act_A)
# Opt = tf.train.RMSPropOptimizer(learning_rate=1E-4, momentum=.8, centered=True).minimize(PL)
Opt = tf.train.MomentumOptimizer(learning_rate=1E-5, momentum=.8).minimize(PL)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.graph.finalize() 

episode = 0
while(1):
# for episode in range(EPISODE):    
    episode += 1
    act_h = [ [0. for j in range(6)] for i in range(action_memo)]
    S_h = [ [ [ [0 for i in range(3)] for j in range(160)] for k in range(210)] for l in range(action_memo)]  # [None, 210, 160, 3]


    S = env.reset() #(210, 160, 3)
    # [env.step(2) for i in range(75 + np.random.randint(10))] # adjust the initial position of the agent
    [env.step(0) for i in range(75 + np.random.randint(10))] # skip the blank render
    Clives = 3
    Reward_cnt = 0.
    CuReward = 0.
    R_list, S_list = [],[]

    
    steps = 0
    if (np.random.random() >= EPSILONE/np.clip(episode-WARMING_EPI,1E-9,None)) and (WARMING_EPI < episode):
        Greedy_flag = False 
    else:
        Greedy_flag = True 
    pass
    while(1):
        steps += 1
    # for step in range(STEP_LIMIT):
        env.render() # show the windows. If you don't need to monitor the state, just comment this.
        # print(S)
        
        # A = env.action_space.sample() # random sampling the actions
        # print(A)

        # sampling action from Q
        # epsilon greedy
        if Greedy_flag or (np.random.random() < .05):
        # if False:
            A = np.random.randint(6)
        else:
            A = sess.run(Act_sample, feed_dict={
                                                Act_S: np.array(S).reshape([-1, 210, 160, 3]),
                                                Act_R: np.array(CuReward).reshape([-1]),
                                                Act_m: np.array(act_h).reshape([-1, action_memo, 6]),
                                                Sta_m: np.array(S_h),
                                                Actions4Act:np.array([0]).reshape([-1])
                                                }
                        )
                                                                              
        pass
        # print(A) # monitor the action
        
        # handling the state-action recorders
        A_oh  = [0. for i in range(6)]
        A_oh[A] = 1.
        act_h = [A_oh] + act_h
        act_h.pop()
        S_h = [S.copy()] + S_h
        S_h.pop()

        Sp = S.copy()
        S, R, finish_flag, info = env.step(A)

        Reward_cnt += R
        if CuReward > REWARD_NORMA:
            REWARD_NORMA = CuReward
        # else:
        #     REWARD_NORMA = (REWARD_NORMA + CuReward)/2
        pass

        # check the action history
        KL_A = KL_div_with_normal(np.array(act_h))
        # print(KL_A)

        # CuReward = CuReward * GAMMA + R
        CuReward = CuReward * GAMMA + (R - REWARD_b) - KL_A * .1
        

        # print('Reward:{}'.format(R)) # the reward will give this action will get how much scores. it's descreted.
        # print('Info:{}'.format(info['ale.lives'])) # info in space invader will give the lives of the current state

        if finish_flag or (Clives > info['ale.lives']):
            Clives = info['ale.lives']
            
            # CuReward = np.clip(CuReward, 0, None)
            # print('This episode is finished ...')

            ## panelize the death state and all the states would cause to die
            S_h.reverse()
            act_h.reverse()
            for death_rec in range(action_memo):
                CuReward -= DIE_PANELTY
                Sp = S_h.pop()
                S_h = [ [ [ [0 for i in range(3)] for j in range(160)] for k in range(210)] ] + S_h
                A = np.argmax(act_h.pop())
                act_h = [[0. for i in range(6)]] + act_h
                sess.run(Opt, 
                        feed_dict={
                                    Act_S: np.array(Sp).reshape([-1, 210, 160, 3]),
                                    Act_R: np.array(CuReward).reshape([-1]),
                                    Act_m: np.array(act_h).reshape([-1, action_memo, 6]),
                                    Sta_m: np.array(S_h),
                                    Actions4Act:np.array(A).reshape([-1])
                                }
                        )
            pass

            if finish_flag:
                break
            else:
                # [env.step(2) for i in range(45 + np.random.randint(10))] # adjust the initial position of the agent
                [env.step(0) for i in range(40)] # skip the blank render
                continue
            pass
        pass 

        # TD
        Loss, _ = sess.run([PL, Opt], 
                            feed_dict={
                                       Act_S: np.array(Sp).reshape([-1, 210, 160, 3]),
                                       Act_R: np.array(CuReward).reshape([-1]),
                                       Act_m: np.array(act_h).reshape([-1, action_memo, 6]),
                                       Sta_m: np.array(S_h),
                                       Actions4Act:np.array(A).reshape([-1])
                                      }
                           )
        print('Action:{}  Loss:{} Epsilon:{} greedy:{}'.format(A, Loss, EPSILONE/np.clip(episode-WARMING_EPI,1E-9,None), Greedy_flag))

    pass
    print("Epi:{}  Score:{}  Loss:{} greedy:{}".format(episode,Reward_cnt,Loss,Greedy_flag))


pass
