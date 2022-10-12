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

np.random.seed(123)
tf.random.set_seed(123)

if __name__ == '__main__':
    env = gym.make("LunarLander-v2", continuous=False, render_mode='human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("state {}, action {}".format(obs_dim, act_dim))

    hidden_sizes = [256,256,256]
    actor_lr = 1e-4
    critic_lr = 2e-4
    target_kl = 0.01
    agent = VPG(obs_dim, act_dim, hidden_sizes, actor_lr, critic_lr, target_kl)

    total_epochs = 100
    steps_per_epoch = 4000
    buffer = ReplayBuffer(obs_dim,act_dim,capacity=steps_per_epoch,gamma=0.99,lamda=0.97)

    logDir = 'logs/vpg' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    ep_ret, ep_len = 0, 0
    state = env.reset()
    for epoch in range(total_epochs):
        sum_ret, sum_len, num_episodes = 0, 0, 0
        for t in range(steps_per_epoch):
            a, logp, value = agent.policy(state[0])
            new_state = env.step(tf.squeeze(a).numpy())
            r, done = new_state[1], new_state[2]
            buffer.store(state[0],a,r,value,logp)
            ep_len += 1
            ep_ret += r
            state = new_state
            if done or (t == steps_per_epoch - 1):
                last_value = 0
                if not done:
                    last_value = tf.squeeze(agent.q(tf.expand_dims(tf.convert_to_tensor(state[0]), 0)))
                buffer.finish_trajectory(last_value)
                sum_ret += ep_ret
                sum_len += ep_len
                num_episodes += 1
                state = env.reset()
                ep_ret, ep_len = 0, 0

        agent.learn(buffer)
        print("Epoch {}, Mean Return {:.4f}, Mean Length {:.4f}".format(
            epoch+1, sum_ret/num_episodes, sum_len/num_episodes))

    env.close()
