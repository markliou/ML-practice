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

def Q(S, AH, SH, Li, reuse=False, trainable = True):
    
    S = (tf.cast(S,tf.float32)-128)/128.  #(210, 160, 3)
    SH = tf.unstack(tf.map_fn(lambda x: tf.stack([tf.concat(tf.unstack(x, axis=0), axis=-1)], axis=0), SH), axis=1)[0]
    SH = (tf.cast(SH,tf.float32)-128)/128.
    AH = tf.reshape(AH, [-1, AH.shape[1] * AH.shape[2]]) #[None, 32, 6]
    AH = AH * 2 - 1
    Li -= 2
    Li = tf.reshape(Li,[-1, 1])

    conv1 = conv2d(tf.concat([S, SH], axis=-1), stride_no=2, reuse=reuse, trainable = trainable, name='conv1') #(105, 80)
    conv1a = conv2d(conv1, stride_no=1, reuse=reuse, trainable = trainable, name='conv1a') #(27, 20)
    conv1b = conv2d(conv1a, stride_no=1, reuse=reuse, trainable = trainable, name='conv1b') #(27, 20)
    conv1c = conv2d(conv1b, stride_no=1, reuse=reuse, trainable = trainable, name='conv1c') #(27, 20)
    conv2 = conv2d(conv1c, stride_no=2, reuse=reuse, trainable = trainable, name='conv2') #(53, 40)
    conv2a = conv2d(conv2, stride_no=1, reuse=reuse, trainable = trainable, name='conv2a') #(27, 20)
    conv2b = conv2d(conv2a, stride_no=1, reuse=reuse, trainable = trainable, name='conv2b') #(27, 20)
    conv2c = conv2d(conv2b, stride_no=1, reuse=reuse, trainable = trainable, name='conv2c') #(27, 20)
    conv2e = conv2d(conv2c, stride_no=1, reuse=reuse, trainable = trainable, name='conv2d') #(27, 20)
    conv2f = conv2d(conv2e, stride_no=1, reuse=reuse, trainable = trainable, name='conv2e') #(27, 20)
    conv3 = conv2d(conv2f, stride_no=2, reuse=reuse, trainable = trainable, name='conv3') #(27, 20)
    conv3a = conv2d(conv3, stride_no=1, reuse=reuse, trainable = trainable, name='conv3a') #(27, 20)
    conv3b = conv2d(conv3a, stride_no=1, reuse=reuse, trainable = trainable, name='conv3b') #(27, 20)
    conv4 = conv2d(conv3b, stride_no=2, reuse=reuse, trainable = trainable ,name='conv4') #(14, 10)
    conv4a = conv2d(conv4, stride_no=1, reuse=reuse, trainable = trainable ,name='conv4a') #(14, 10)
    conv4b = conv2d(conv4a, stride_no=1, reuse=reuse, trainable = trainable ,name='conv4b') #(14, 10)
    conv5 = conv2d(conv4b, stride_no=2, reuse=reuse, trainable = trainable, name='conv5') #(7, 5)
    conv5a = conv2d(conv5, stride_no=1, reuse=reuse, trainable = trainable, name='conv5a') #(7, 5)
    conv5b = conv2d(conv5a, stride_no=1, reuse=reuse, trainable = trainable, name='conv5b') #(7, 5)
    conv6 = conv2d(conv5b, stride_no=2, reuse=reuse, trainable = trainable, name='conv6') #(4, 3)
    conv6a = conv2d(conv6, stride_no=1, reuse=reuse, trainable = trainable, name='conv6a') #(4, 3)
    conv6b = conv2d(conv6a, stride_no=1, reuse=reuse, trainable = trainable, name='conv6b') #(4, 3)
    
    f1 = tf.layers.flatten(conv6b)
    f2 = tf.layers.dense(f1, 512, activation=tf.nn.elu, trainable = trainable, reuse=reuse, name='f2')
    f3 = tf.layers.dense(f2, 256, activation=tf.nn.elu, trainable = trainable, reuse=reuse, name='f3')
    f4 = tf.layers.dense(f3, 128, activation=tf.nn.elu, trainable = trainable, reuse=reuse, name='f4')
    f5 = tf.layers.dense(f4, 64, activation=tf.nn.elu, trainable = trainable, reuse=reuse, name='f5')
    f6 = tf.layers.dense(f5, 32, activation=tf.nn.elu, trainable = trainable, reuse=reuse, name='f6')
    f7 = tf.layers.dense(f6, 64, activation=tf.nn.elu, trainable = trainable, reuse=reuse, name='f7')
    
    fo = tf.layers.dense(f7, 32, activation=tf.nn.elu, trainable = trainable, reuse=reuse, name='fo')
    out = tf.layers.dense(fo, 6, trainable = trainable, reuse=reuse, name='out')

    return out
pass

def KL_div_with_normal(X):
    ## X with the shape of (n, l)
    X = np.sum(X, axis=0)/X.shape[0]
    return np.clip(np.sum(X * (np.log(X+1E-20) - np.log(1/X.shape[0]))), 0, None)
pass

def bullet_avoidence(S):
    # if S[-16].mean() >= 4.27:
    #     return 1
    # else:
    #     return 0
    # pass
    return S[-16].mean() - 4.27
pass

# Enviroment settings
STEP_LIMIT = 1000
EPISODE = 1000
EPSILONE = 1.
REWARD_b = .5 
REWARD_NORMA = 50 # because the peak reward is close to 500, empiritically
STEP_NORMA = 1
GAMMA = .98
DIE_PANELTY = REWARD_NORMA
WARMING_EPI = 0
UPDATE = 0
OUTPUTFILE = 'rec.txt'

env = gym.make('SpaceInvaders-v0') 
OutFile = open(OUTPUTFILE, 'w', buffering=1)

# Actor settings
action_memo = 32
score_delay = 4
td_batch = 32
Act_S = tf.placeholder(tf.int8, [None, 210, 160, 3])
Act_R = tf.placeholder(tf.float32, [None])
Act_m = tf.placeholder(tf.float32, [None, action_memo, 6])
Sta_m = tf.placeholder(tf.int8, [None, action_memo, 210, 160, 3])
RewardNorma = tf.placeholder(tf.float32)
Actions4Act = tf.placeholder(tf.uint8, [None], name='Actions4Act')
Actions4Act_oh = tf.one_hot(Actions4Act, 6) 
Act_clive = tf.placeholder(tf.float32, [None])
Loss = 0

Act_A = Q(Act_S, Act_m, Sta_m, Act_clive, trainable = True, reuse=False) # give the current state, action history and state history
Act_sample = tf.argmax(tf.nn.softmax(Act_A), axis=-1)[0]

# PL = Act_R * -tf.log(tf.reduce_sum(tf.nn.softmax(Act_A) * Actions4Act_oh)+1E-9)
PL = (Act_R*2 /RewardNorma) * tf.nn.softmax_cross_entropy_with_logits_v2(labels=Actions4Act_oh, logits=Act_A)
PL = tf.reduce_mean(PL)
Opt = tf.train.RMSPropOptimizer(learning_rate=1E-6, momentum=.8, centered=True).minimize(PL)
# Opt = tf.contrib.opt.AdamWOptimizer(1E-4, 1E-5).minimize(PL)
# Opt = tf.train.MomentumOptimizer(learning_rate=1E-4, momentum=.8).minimize(PL)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.graph.finalize() 

episode = 0
# create TD tuple
td_cnt = 0
TD_S = [[[[0 for i in range(3)] for j in range(160)] for k in range(210)] for l in range(td_batch)]                   # Act_S: np.array(S).reshape([-1, 210, 160, 3]),
TD_Act_m = [[[0 for i in range(6)] for j in range(action_memo)] for l in range(td_batch)]         # Act_m: np.array(act_hp).reshape([-1, action_memo, 6]),
TD_S_hp = [[[[[0 for i in range(3)] for j in range(160)] for k in range(210)] for m in range(action_memo)] for l in range(td_batch)]      # Sta_m: np.array(S_hp).reshape([-1, action_memo, 210, 160, 3]),
TD_Clives = [3 for i in range(td_batch)]        # Act_clive: np.array([Clives]),
TD_Actions4Act = [1 for i in range(td_batch)]        # Actions4Act:np.array([0]).reshape([-1])
TD_Reward = [0 for i in range(td_batch)]

while(1):
# for episode in range(EPISODE):    
    episode += 1
    act_h = [ [0. for j in range(6)] for i in range(score_delay+action_memo)]
    S_h = [ [ [ [0 for i in range(3)] for j in range(160)] for k in range(210)] for l in range(score_delay+action_memo)]  # [None, 210, 160, 3]
    delay_CuReward = [0. for i in range(score_delay)]

    S = env.reset() #(210, 160, 3)
    # [env.step(2) for i in range(75 + np.random.randint(10))] # adjust the initial position of the agent
    # [env.step(0) for i in range(75 + np.random.randint(10))] # skip the blank render
    Clives = 3
    Reward_cnt = 0.
    CuReward = 0.
    BA = 0.
    R_list, S_list = [],[]

    
    steps = 0
    if (np.random.random() >= EPSILONE/np.clip(episode-WARMING_EPI,1E-9,None)) and (WARMING_EPI < episode):
        Greedy_flag = False 
    else:
        Greedy_flag = True 
    pass
    while(1):
        steps += 1
        if STEP_NORMA < steps:
            STEP_NORMA = (steps + STEP_NORMA)/2
        pass 
    # for step in range(STEP_LIMIT):
        # env.render() # show the windows. If you don't need to monitor the state, just comment this.
        # print(S)
        
        # A = env.action_space.sample() # random sampling the actions
        # print(A)

        # sampling action from Q
        # epsilon greedy
        # if Greedy_flag or (np.random.random() < .05):
        # if np.random.random() < .05:
        if np.random.random() < (1/td_batch) :
        # if False:
            A = np.random.randint(6)
        else:
            
            S_hp = S_h[0:action_memo].copy()
            Ap = act_h[score_delay].copy()
            act_hp = act_h[0:action_memo].copy()

            A = sess.run(Act_sample, feed_dict={
                                                Act_S: np.array(S).reshape([-1, 210, 160, 3]),
                                                Act_m: np.array(act_hp).reshape([-1, action_memo, 6]),
                                                Sta_m: np.array(S_hp).reshape([-1, action_memo, 210, 160, 3]),
                                                Act_clive: np.array([Clives]),
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
        BA = bullet_avoidence(S) # bullet avoidence
        # if CuReward > REWARD_NORMA:
        #     REWARD_NORMA = CuReward
        # pass

        # check the action history
        KL_A = KL_div_with_normal(np.array(act_h))
        # print(KL_A)

        # CuReward = CuReward * GAMMA + R
        # CuReward = CuReward * GAMMA + (R - REWARD_b) - KL_A + Reward_cnt/steps
        # CuReward = CuReward * GAMMA + (R + BA * 1.5 - REWARD_b * (3 - Clives)) + (steps/STEP_NORMA) * (3 - Clives) - KL_A * .5
        CuReward = CuReward * GAMMA + (R + BA * 1.5 - REWARD_b * (3 - Clives)) - KL_A * .3
        # CuReward = CuReward * GAMMA + (R - REWARD_b)
        # CuReward = R - REWARD_b 
        

        # print('Reward:{}'.format(R)) # the reward will give this action will get how much scores. it's descreted.
        # print('Info:{}'.format(info['ale.lives'])) # info in space invader will give the lives of the current state

        if finish_flag or (Clives > info['ale.lives']):
        # if False:
            Clives = info['ale.lives']
            
            # CuReward = np.clip(CuReward, 0, None)
            # print('This episode is finished ...')

            # ### panelize the death state and all the states would cause to die
            Sp = S_h[score_delay].copy()
            S_hp = S_h[score_delay : score_delay+action_memo].copy()
            Ap = np.argmax(act_h[score_delay].copy())
            act_hp = act_h[score_delay : score_delay+action_memo].copy()

            for i in range(0):
                Loss, _ = sess.run([PL, Opt], 
                                    feed_dict={
                                            Act_S: np.array(Sp).reshape([-1, 210, 160, 3]),
                                            Act_R: np.array(-DIE_PANELTY).reshape([-1]),
                                            Act_m: np.array(act_hp).reshape([-1, action_memo, 6]),
                                            Sta_m: np.array(S_hp).reshape([-1, action_memo, 210, 160, 3]),
                                            Act_clive: np.array([Clives]).reshape([-1]),
                                            RewardNorma:REWARD_NORMA,
                                            Actions4Act:np.array(Ap).reshape([-1])
                                            }
                                )
            pass

            if finish_flag:
                # clear the TD recorders
                
                # rebuilding the whole recs seems make program slow, so using the pushing method to instead
                # TD_S = [[[[0. for i in range(3)] for j in range(160)] for k in range(210)] for l in range(td_batch)]                   # Act_S: np.array(S).reshape([-1, 210, 160, 3]),
                # TD_Act_m = [[[0. for i in range(6)] for j in range(action_memo)] for l in range(td_batch)]         # Act_m: np.array(act_hp).reshape([-1, action_memo, 6]),
                # TD_S_hp = [[[[[0. for i in range(3)] for j in range(160)] for k in range(210)] for m in range(action_memo)] for l in range(td_batch)]      # Sta_m: np.array(S_hp).reshape([-1, action_memo, 210, 160, 3]),
                # TD_Clives = [3. for i in range(td_batch)]        # Act_clive: np.array([Clives]),
                # TD_Actions4Act = [0. for i in range(td_batch)]
                # TD_Reward = [0 for i in range(td_batch)]
                # td_cnt = 0

                for i in range(td_batch):
                    TD_S.pop()
                    TD_Reward.pop()
                    TD_Act_m.pop()
                    TD_S_hp.pop()
                    TD_Clives.pop()
                    TD_Actions4Act.pop()
                    
                    TD_S = [[[[0. for i in range(3)] for j in range(160)] for k in range(210)]] + TD_S
                    TD_Reward = [0] + TD_Reward
                    TD_Act_m = [[[0. for i in range(6)] for j in range(action_memo)]] + TD_Act_m
                    TD_S_hp = [[[[[0. for i in range(3)] for j in range(160)] for k in range(210)] for m in range(action_memo)]] + TD_S_hp
                    TD_Clives = [3] + TD_Clives
                    TD_Actions4Act = [0] + TD_Actions4Act
                pass


                break
            else:
                # [env.step(2) for i in range(45 + np.random.randint(10))] # adjust the initial position of the agent
                # [env.step(0) for i in range(40)] # skip the blank render
                continue
            pass

             

        else: 

            #### TD

            # score delay
            Sp = S_h[score_delay].copy()
            S_hp = S_h[score_delay : score_delay+action_memo].copy()
            Ap = np.argmax(act_h[score_delay].copy())
            act_hp = act_h[score_delay : score_delay+action_memo].copy()
            delay_CuReward.pop() 
            delay_CuReward = [CuReward] + delay_CuReward
            if np.sum(delay_CuReward) > REWARD_NORMA:
                REWARD_NORMA = (np.sum(delay_CuReward) + REWARD_NORMA)/2
            pass

            td_cnt += 1
            if td_cnt == td_batch:
                
                Loss, _ = sess.run([PL, Opt], 
                                    feed_dict={
                                            Act_S: np.array(TD_S).reshape([-1, 210, 160, 3]),
                                            Act_R: np.array(TD_Reward),
                                            Act_m: np.array(TD_Act_m).reshape([-1, action_memo, 6]),
                                            Sta_m: np.array(TD_S_hp).reshape([-1, action_memo, 210, 160, 3]),
                                            Act_clive: np.array([TD_Clives]).reshape([-1]),
                                            RewardNorma:REWARD_NORMA,
                                            Actions4Act:np.array(TD_Actions4Act).reshape([-1])
                                            }
                                )
                UPDATE += 1
                
                # TD_S = [[[[0 for i in range(3)] for j in range(160)] for k in range(210)] for l in range(td_batch)]                   # Act_S: np.array(S).reshape([-1, 210, 160, 3]),
                # TD_Act_m = [[[0 for i in range(6)] for j in range(action_memo)] for l in range(td_batch)]         # Act_m: np.array(act_hp).reshape([-1, action_memo, 6]),
                # TD_S_hp = [[[[[0 for i in range(3)] for j in range(160)] for k in range(210)] for m in range(action_memo)] for l in range(td_batch)]      # Sta_m: np.array(S_hp).reshape([-1, action_memo, 210, 160, 3]),
                # TD_Clives = [3 for i in range(td_batch)]        # Act_clive: np.array([Clives]),
                # TD_Actions4Act = [0 for i in range(td_batch)] 
                td_cnt = 0 
            else:
                TD_S.pop()
                TD_Reward.pop()
                TD_Act_m.pop()
                TD_S_hp.pop()
                TD_Clives.pop()
                TD_Actions4Act.pop()
                
                TD_S = [Sp] + TD_S
                TD_Reward = [np.clip(np.sum(delay_CuReward), -REWARD_NORMA * .1, None)] + TD_Reward
                TD_Act_m = [act_hp] + TD_Act_m
                TD_S_hp = [S_hp] + TD_S_hp
                TD_Clives = [Clives] + TD_Clives
                TD_Actions4Act = [Ap] + TD_Actions4Act
                
            pass
            print('Action:{}  Loss:{} Epsilon:{} greedy:{} Score:{} Update:{}'.format(A, Loss, EPSILONE/np.clip(episode-WARMING_EPI,1E-9,None), Greedy_flag, Reward_cnt, UPDATE))
            
        pass

    pass
    print("Epi:{}  Score:{}  Loss:{} greedy:{}".format(episode,Reward_cnt,Loss,Greedy_flag))
    OutFile.write("Epi:{}  Score:{}  Loss:{} greedy:{}\n".format(episode,Reward_cnt,Loss,Greedy_flag))


pass
