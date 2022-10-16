import sys
sys.path.append('..')
sys.path.append('.')
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime
from agent.ppo import PPO, ReplayBuffer

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
    env = gym.make("CartPole-v1", render_mode='human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("state {}, action {}".format(obs_dim, act_dim))

    hidden_sizes = [64,64]
    clip_ratio = 0.2
    actor_lr = 1e-4
    critic_lr = 2e-4
    beta = 0.001
    agent = PPO(obs_dim, act_dim, hidden_sizes, clip_ratio, actor_lr, critic_lr, beta)

    buffer = ReplayBuffer(obs_dim,act_dim,capacity=3000,gamma=0.99,lamda=0.97)

    logDir = 'logs/ppo' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    t, update_steps = 0, 2000
    total_episodes, ep_max_step = 1000, 500
    ep_ret_list, avg_ret_list = [], []
    for ep in range(total_episodes):
        done, ep_ret, ep_step = False, 0, 0
        state = env.reset()
        while not done and ep_step < ep_max_step:
            a, prob, value = agent.policy(state[0])
            new_state = env.step(a)
            r, done = new_state[1], new_state[2]
            buffer.store(state[0],tf.one_hot(a,act_dim).numpy(),r,value,prob)
            t += 1
            ep_step += 1
            ep_ret += r
            state = new_state

        last_value = 0 if done else agent.value(state[0])
        buffer.finish_trajectory(last_value)

        if buffer.ptr > update_steps or (ep + 1) == total_episodes:
            agent.learn(buffer, actor_iter=120, critic_iter=120)

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
