import sys
sys.path.append('..')
sys.path.append('.')
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime
from agent.core import ReplayBuffer_P
from agent.ppo import PPO

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

    hidden_sizes = [256,256,64]
    clip_ratio = 0.2
    actor_lr = 1e-4
    critic_lr = 2e-4
    target_kl = 0.01
    agent = PPO(obs_dim, act_dim, hidden_sizes, clip_ratio, actor_lr, critic_lr, target_kl)
    buffer = ReplayBuffer_P(obs_dim,act_dim,capacity=10000,gamma=0.99,lamda=0.97)

    logDir = 'logs/ppo' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    total_episodes = 1000
    t, max_step, update_steps = 0, 500, 3000
    ep_ret_list, avg_ret_list = [], []
    for ep in range(total_episodes):
        ep_ret, ep_step = 0, 0
        done = False
        state = env.reset()
        o = state[0]
        while not done and ep_step < max_step:
            a, logp, value = agent.policy(o)
            state = env.step(tf.squeeze(a).numpy())
            o2,r,done = state[0],state[1],state[2]
            buffer.store(o,a,r,value,logp)
            t += 1
            ep_step += 1
            ep_ret += r
            o = o2

        last_value = 0 if done else agent.value(o)
        buffer.finish_trajectory(last_value) # finish trajectory if reached to a terminal state

        if buffer.ptr > update_steps:
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
