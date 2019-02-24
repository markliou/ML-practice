# if you use Windows, try this 
# pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py 
# https://zhuanlan.zhihu.com/p/33834336
import gym 


# more note information
# https://blog.techbridge.cc/2017/11/04/openai-gym-intro-and-q-learning/

# space invader has 6 actions
env = gym.make('SpaceInvaders-v0') 
S = env.reset()

for step in range(1000):
    # env.render() # show the windows. If you don't need to monitor the state, just comment this.
    # print(S)
    A = env.action_space.sample()
    print(A)
    S, R, finish_flag, info = env.step(A)
    print('Reward:{}'.format(R)) # the reward will give this action will get how much scores. it's descreted.
    print('Info:{}'.format(info)) # info in space invader will give the lives of the current state
    if finish_flag:
        print('This episode is finished ...')
        break