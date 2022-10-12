"""
reference:
https://keras.io/examples/rl/ddpg_pendulum/
"""
import sys
sys.path.append('..')
sys.path.append('.')
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
from agent.core import OUNoise
from agent.ddpg import DDPG, ReplayBuffer

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
    env = gym.make("Pendulum-v1", render_mode = 'human')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    print("state {}, action {}, limit {}".format(obs_dim,act_dim,act_limit))

    buffer = ReplayBuffer(obs_dim,act_dim,capacity=50000,batch_size=64)
    noise = OUNoise(mu=np.zeros(act_dim),sigma=float(0.2)*np.ones(act_dim))
    hidden_sizes = [256,256]
    gamma = 0.99
    polyak = 0.995
    pi_lr = 1e-3
    q_lr = 2e-3
    agent = DDPG(obs_dim,act_dim,hidden_sizes,act_limit,gamma,polyak,pi_lr,q_lr,noise)

    logDir = 'logs/ddpg' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    total_episodes, max_steps = 1000, 200
    ep_ret_list, avg_ret_list = [], []
    for ep in range(total_episodes):
        done, ep_ret, ep_step = False, 0, 0
        state = env.reset()
        while not done and ep_step < max_steps:
            a = agent.policy(state[0])
            new_state = env.step(a)
            r, done = new_state[1], new_state[2]
            buffer.store(state[0],a,r,new_state[0],done)
            ep_step += 1
            ep_ret += r
            state = new_state

            agent.learn(buffer)

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        ep_ret_list.append(ep_ret)
        avg_ret = np.mean(ep_ret_list[-40:])
        avg_ret_list.append(avg_ret)
        print("Episode *{}* average reward is {}".format(ep, avg_ret))

    env.close()

    plt.plot(avg_ret_list)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Episodic Reward')
    plt.show()
