import sys
sys.path.append('..')
sys.path.append('.')
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime
from agent.vpg import VPG, ReplayBuffer

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
Limiting GPU memory growth
"""
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

if __name__ == '__main__':
    logDir = 'logs/vpg' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    env = gym.make("CartPole-v1", render_mode='human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("state {}, action {}".format(obs_dim, act_dim))

    buffer = ReplayBuffer(obs_dim,act_dim,capacity=50000,gamma=0.99,lamda=0.97)
    agent = VPG(obs_dim,act_dim,hidden_sizes=[32,32],pi_lr=1e-4,q_lr=2e-4,target_kld=1e-2)

    ep_ret_list, avg_ret_list = [], []
    t, update_after = 0, 1e3
    total_episodes, max_ep_steps = 500, 500
    for ep in range(total_episodes):
        done, ep_ret, step = False, 0, 0
        state = env.reset()
        while not done and step < max_ep_steps:
            a, logp = agent.policy(state[0])
            value = agent.value(state[0])
            new_state = env.step(a)
            r, done = new_state[1], new_state[2]
            buffer.store(state[0],a,r,value,logp)
            state = new_state
            ep_ret += r
            step += 1
            t += 1

        last_value = 0 if done else agent.value(state[0])
        buffer.finish_trajectory(last_value)
        if buffer.ptr > update_after:
            agent.learn(buffer)

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        ep_ret_list.append(ep_ret)
        avg_ret = np.mean(ep_ret_list[-40:])
        avg_ret_list.append(avg_ret)
        print("Episode *{}* average reward is {:.4f}, episode length {}".format(ep, avg_ret, step))

    env.close()

    plt.plot(avg_ret_list)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Episodic Reward')
    plt.show()
