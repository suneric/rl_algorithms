import sys
sys.path.append('..')
sys.path.append('.')
import gym
import numpy as np
import tensorflow as tf
import os
import datetime
from core.dqn import *
import matplotlib.pyplot as plt

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
Limiting GPU memory growth
"""
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

np.random.seed(123)

if __name__ == '__main__':
    env = gym.make('CartPole-v0', render_mode='human')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    buffer = ReplayBuffer(obs_dim,act_dim,size=50000,batch_size=64)
    hidden_sizes = [64,64]
    gamma = 0.99
    lr = 1e-3
    agent = DQN(obs_dim,act_dim,hidden_sizes,gamma,lr)

    logDir = 'logs/dqn' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    epsilon, epsilon_stop = 0.99, 0.1
    decay = 0.999
    t, max_steps, sync_step = 0, 200, 50
    total_episodes = 1000
    ep_ret_list, avg_ret_list = [], []
    for ep in range(total_episodes):
        epsilon = max(epsilon_stop, epsilon*decay)
        ep_ret, avg_ret = 0, 0
        state = env.reset()
        o = state[0]
        for _ in range(max_steps): # an episode
            a = agent.policy(o,epsilon)
            print(a)
            state = env.step(a)
            o2,r,d = state[0],state[1],state[2]
            buffer.store(o,a,r,o2,d)
            agent.learn(buffer)
            o = o2
            ep_ret += r
            t += 1
            if t % sync_step == 0:
                agent.update_stable()

            if done:
                break

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        ep_ret_list.append(ep_ret)
        avg_ret = np.mean(ep_ret_list[-40:])
        avg_ret_list.append(avg_ret)
        print("Episode *{}* average reward is {}, total steps {}".format(ep, avg_ret, t))

    env.close()

    plt.plot(avg_ret_list)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Episodic Reward')
    plt.show()
