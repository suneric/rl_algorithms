import sys
sys.path.append('..')
sys.path.append('.')
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime
from agent.core import ReplayBuffer_Q
from agent.dqn import DQN

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
Limiting GPU memory growth
"""
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

np.random.seed(123)

if __name__ == '__main__':
    env = gym.make("LunarLander-v2", continuous = False, render_mode='human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("state {}, action {}".format(obs_dim,act_dim))

    buffer = ReplayBuffer_Q(obs_dim,act_dim,capacity=50000,batch_size=64,continuous=False)
    hidden_sizes = [256,256,64]
    gamma = 0.99
    lr = 1e-4
    agent = DQN(obs_dim,act_dim,hidden_sizes,gamma,lr)

    logDir = 'logs/dqn' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    total_episodes = 1000
    epsilon, epsilon_stop, decay = 0.99, 0.1, 0.995
    t, update_after, sync_steps = 0, 1000, 100
    ep_ret_list, avg_ret_list = [], []
    for ep in range(total_episodes):
        epsilon = max(epsilon_stop, epsilon*decay)
        ep_ret, avg_ret = 0, 0
        done = False
        state = env.reset()
        o = state[0]
        while not done:
            a = agent.policy(o,epsilon)
            state = env.step(a)
            o2,r,done = state[0],state[1],state[2]
            buffer.store(o,a,r,o2,done)
            ep_ret += r
            t += 1
            o = o2

            if t > update_after:
                agent.learn(buffer)
            if t % sync_steps == 0:
                agent.update_stable()

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        ep_ret_list.append(ep_ret)
        avg_ret = np.mean(ep_ret_list[-40:])
        avg_ret_list.append(avg_ret)
        print("Episode *{}* average reward is {}, total steps {}, epsilon {}".format(ep, avg_ret, t, epsilon))

    env.close()

    plt.plot(avg_ret_list)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Episodic Reward')
    plt.show()
