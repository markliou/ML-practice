import gym 
import tensorflow as tf
import numpy as np
import os

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
    conv5 = conv2d(conv4, 256, stride_no=1) #(7, 5)
    #conv5 = tf.layers.Dropout(.5)(conv5)
    conv6 = conv2d(conv5, 512, stride_no=1) #(4, 3)
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
ALPHA = .6
DIE_PANELTY = 0
WARMING_EPI = 0
BEST_REC = 0.
BEST_STEPS = 1.
STATE_GAMMA = .6

env = gym.make('SpaceInvaders-v0') 
os.system("echo > score_rec.txt") #clean the previoud recorders

# Actor settings
Opt_size = 16 # skip frames
OPT_FLAG = False
Act_S = tf.placeholder(tf.int8, [None, 210, 160, 3])
Act_Sp = tf.placeholder(tf.int8, [None, 210, 160, 3])
Act_R = tf.placeholder(tf.float32, [None])
Actions4Act = tf.placeholder(tf.uint8, [None])
Actions4Act_oh = tf.one_hot(Actions4Act, 4) 

Act_A = Q(Act_S)
Command_A = tf.argmax(Act_A, axis=-1)

Act_Ap = Q(Act_Sp)
PL = tf.pow((Act_R + tf.reduce_max(Act_A) - tf.reduce_max(Act_Ap * Actions4Act_oh)), 2) #Q 

#Opt = tf.train.RMSPropOptimizer(1E-4, momentum=.0, centered=True).minimize(PL)
#Opt = tf.train.MomentumOptimizer(learning_rate=1E-6, momentum=.8).minimize(PL)

optimizer = tf.train.RMSPropOptimizer(1E-3, .6, momentum=.0, centered=True)
gr, va = zip(*optimizer.compute_gradients(PL))
gr = [None if gr is None else tf.clip_by_norm(grad, 5.) for grad in gr]
Opt = optimizer.apply_gradients(zip(gr, va))

sess = tf.Session()
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

        Sp = S.copy()
        S, R, finish_flag, info = env.step(A)
        GameScore += R
        S = Sp * STATE_GAMMA * .5 + S * .5  # keep the previoud state as input would be creating a RNN like condition
       
        # handling the reward and actions
        if R > 0:
            OPT_FLAG = True
        pass
        
        if A in [0]:
            R += REWARD_b * .5 # give the reward for moving. This would be helpful for telling agent to avopod bullet
        elif A in [2,3]:
            R += REWARD_b * .1
        elif A in [1]:
            Shooting_S.append([Sp, S]) # s, s'. Treat it as MC and memory replay hybrid
        pass

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
            CuReward = -1
            # CuReward = np.clip(CuReward, 0, None)
            # print('This episode is finished ...')
            A = sess.run(Command_A, feed_dict={Act_S:np.array(Sp).reshape([1, 210, 160, 3])})[0]
            Loss, _ = sess.run([PL, Opt], 
                               feed_dict={
                                          Act_S:np.array(S).reshape([-1, 210, 160, 3]),
                                          Act_Sp:np.array(Sp).reshape([-1, 210, 160, 3]),
                                          Act_R:np.array(CuReward).reshape([-1]),
                                          Actions4Act:np.array(A).reshape([-1]) 
                                         }
                               )
            if finish_flag:
                if BEST_REC < GameScore:
                    BEST_REC = GameScore 
                pass
                if BEST_STEPS < steps:
                    BEST_STEPS = steps 
                pass
                if REWARD_b < (GameScore/steps) :
                    REWARD_b = (GameScore/steps)
                pass
                #if REWARD_b < (CuReward/steps):
                #    REWARD_b = CuReward/steps
                #pass
                if DIE_PANELTY < CuReward:
                    DIE_PANELTY = CuReward * .8 
                    #print(DIE_PANELTY)
                pass
                os.system("echo {} >> score_rec.txt".format(GameScore))
                break
            else:
                continue
            pass
        pass 
        # TD
        if  OPT_FLAG and len(Shooting_S) > 0: # shooting MC
            SN = len(Shooting_S)
            #print('SN {}'.format(SN))
            SR = (R * 10 )/SN
            #print('SR {}'.format(SR))
            for Si, Spi in Shooting_S + Shooting_S:
                Loss, _ = sess.run([PL, Opt], 
                                    feed_dict={
                                            Act_S:np.array(S).reshape([-1, 210, 160, 3]),
                                            Act_Sp:np.array(Sp).reshape([-1, 210, 160, 3]),
                                            Act_R:np.array(SR).reshape([-1]),
                                            Actions4Act:np.array(1).reshape([-1])
                                            }
                                   )
            pass
            OPT_FLAG = False
            Shooting_S = []
        elif A in [0, 2, 3] and steps % Opt_size == 0:
            Loss, _ = sess.run([PL, Opt],
                                feed_dict={
                                        Act_S:np.array(S).reshape([-1, 210, 160, 3]),
                                        Act_Sp:np.array(Sp).reshape([-1, 210, 160, 3]),
                                        Act_R:np.array(CuReward).reshape([-1]),
                                        Actions4Act:np.array(A).reshape([-1])
                                          }
                               )
        pass
        #print('Action:{}  Loss:{} Epsilon:{} greedy:{} score:{}'.format(A, Loss, EPSILONE/np.clip(episode-WARMING_EPI,1E-9,None), Greedy_flag, GameScore))
    pass
    print("Epi:{}  Score:{}  Loss:{}  Reward:{}".format(episode,GameScore,Loss,CuReward))

pass
