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
    actor_lr = 3e-4
    critic_lr = 1e-3
    target_kl = 0.01
    agent = PPO(obs_dim, act_dim, hidden_sizes, clip_ratio, actor_lr, critic_lr, target_kl)
    size = 1000
    buffer = ReplayBuffer_P(obs_dim,act_dim,capacity=size,gamma=0.99,lamda=0.97)

    logDir = 'logs/ppo' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    total_epochs = 2000
    t, ep, max_step = 0, 0, size
    ep_ret_list, avg_ret_list = [], []
    for _ in range(total_epochs):
        ep_ret, avg_ret = 0, 0
        state = env.reset()
        o = state[0]
        for i in range(max_step):
            a, logp, value = agent.policy(o)
            state = env.step(a[0].numpy())
            o2,r,done = state[0],state[1],state[2]
            buffer.store(o,a,r,value,logp)
            ep_ret += r
            t += 1
            o = o2
            # finish trajectory if reached to a terminal state
            if done or (i == max_step-1):
                last_value = 0 if done else agent.critic(tf.expand_dims(tf.convert_to_tensor(o), 0))
                buffer.finish_trajectory(last_value)

                with summaryWriter.as_default():
                    tf.summary.scalar('episode reward', ep_ret, step=ep)

                ep_ret_list.append(ep_ret)
                avg_ret = np.mean(ep_ret_list[-40:])
                avg_ret_list.append(avg_ret)
                print("Episode *{}* average reward is {}, total steps {}".format(ep, avg_ret, t))
                ep += 1
                ep_ret, avg_ret = 0, 0
                state = env.reset()
                o = state[0]

        agent.learn(buffer, actor_iter=120, critic_iter=120)

    env.close()

    plt.plot(avg_ret_list)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Episodic Reward')
    plt.show()
