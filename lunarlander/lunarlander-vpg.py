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
from agent.vpg import VPG

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
Limiting GPU memory growth
"""
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

np.random.seed(123)

if __name__ == '__main__':
    env = gym.make("LunarLander-v2", continuous=False, render_mode='human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("state {}, action {}".format(obs_dim, act_dim))

    hidden_sizes = [256,256,64]
    actor_lr = 1e-4
    critic_lr = 2e-4
    target_kl = 0.01
    agent = VPG(obs_dim, act_dim, hidden_sizes, actor_lr, critic_lr, target_kl)

    total_epochs = 30
    steps_per_epoch = 4000
    buffer = ReplayBuffer_P(obs_dim,act_dim,capacity=steps_per_epoch,gamma=0.99,lamda=0.97)

    logDir = 'logs/ppo' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    state = env.reset()
    o = state[0]
    ep_ret, ep_len = 0, 0
    for epoch in range(total_epochs):
        sum_ret, sum_len, num_episodes = 0, 0, 0
        for t in range(steps_per_epoch):
            a, logp, value = agent.policy(o)
            state = env.step(tf.squeeze(a).numpy())
            o2,r,done = state[0],state[1],state[2]
            buffer.store(o,a,r,value,logp)
            ep_len += 1
            ep_ret += r
            o = o2
            if done or (t == steps_per_epoch - 1):
                last_value = 0 if done else agent.value(o)
                buffer.finish_trajectory(last_value)
                sum_ret += ep_ret
                sum_len += ep_len
                num_episodes += 1
                state = env.reset()
                o = state[0]
                ep_ret, ep_len = 0, 0

        agent.learn(buffer)
        print("Epoch {}, Mean Return {:.4f}, Mean Length {:.4f}".format(
            epoch+1, sum_ret/num_episodes, sum_len/num_episodes))

    env.close()
