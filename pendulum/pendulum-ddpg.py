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
    logDir = 'logs/ddpg' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    env = gym.make("Pendulum-v1", render_mode = 'human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    print("state {}, action {}, limit {}".format(obs_dim,act_dim,act_limit))

    noise = OUNoise(mu=np.zeros(act_dim),sigma=float(0.2)*np.ones(act_dim))
    buffer = ReplayBuffer(obs_dim,act_dim,capacity=50000,batch_size=64)
    hidden_sizes=[128,128]
    agent = DDPG(obs_dim,act_dim,hidden_sizes,act_limit,gamma=0.99,polyak=0.995,pi_lr=1e-4,q_lr=2e-4,noise_obj=noise)

    ep_ret_list, avg_ret_list = [], []
    t, start_steps, update_after = 0, 5e3, 1e3
    total_episodes, ep_max_steps = 1000, 500
    for ep in range(total_episodes):
        done, ep_ret, step = False, 0, 0
        state = env.reset()
        while not done and step < ep_max_steps:
            if t > start_steps: # trick for improving exploration
                a = agent.policy(state[0])
            else:
                a = env.action_space.sample()
            new_state = env.step(a)
            r, done = new_state[1], new_state[2]
            buffer.store(state[0],a,r,new_state[0],done)
            state = new_state
            ep_ret += r
            step += 1
            t += 1

            if t > update_after:
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
