import sys
sys.path.append('..')
sys.path.append('.')
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime
from agent.dqn import DQN, ReplayBuffer

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
Limiting GPU memory growth
"""
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

np.random.seed(123)
tf.random.set_seed(123)

if __name__ == '__main__':
    logDir = 'logs/dqn' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    env = gym.make("LunarLander-v2", continuous=False, render_mode='human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("state {}, action {}".format(obs_dim,act_dim))

    buffer = ReplayBuffer(obs_dim,act_dim,capacity=100000,batch_size=128)
    agent = DQN(obs_dim,act_dim,hidden_sizes=[512,512],gamma=0.99,lr=2e-4,update_freq=500)

    ep_ret_list, avg_ret_list = [], []
    epsilon, epsilon_stop, decay = 0.99, 0.1, 0.995
    t, update_after = 0, 2500
    total_episodes, ep_max_steps = 1000, 500
    for ep in range(total_episodes):
        epsilon = max(epsilon_stop, epsilon*decay)
        done, ep_ret, step = False, 0, 0
        state = env.reset()
        while not done and step < ep_max_steps:
            a = agent.policy(state[0], epsilon)
            new_state = env.step(a)
            r, done = new_state[1], new_state[2]
            buffer.store(state[0],a,r,new_state[0],done)
            state = new_state
            ep_ret += r
            step += 1
            t += 1

            if buffer.ptr > update_after:
                agent.learn(buffer)

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
