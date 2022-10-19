import sys
sys.path.append('..')
sys.path.append('.')
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
from agent.core import OUNoise, GSNoise
from agent.td3 import TD3, ReplayBuffer

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
    logDir = 'logs/td3' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    env = gym.make("Pendulum-v1", render_mode = 'human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    print("state {}, action {}, limit {}".format(obs_dim,act_dim,act_limit))

    noise = OUNoise(mu=np.zeros(act_dim),sigma=float(0.2)*np.ones(act_dim))
    buffer = ReplayBuffer(obs_dim,act_dim,capacity=50000,batch_size=64)
    hidden_sizes = [256,256]
    agent = TD3(obs_dim,act_dim,hidden_sizes,act_limit,gamma=0.99,polyak=0.995,pi_lr=1e-3,q_lr=2e-3,noise_obj=noise)

    ep_ret_list, avg_ret_list = [], []
    t, start_steps, update_after = 0, 1000, 500
    total_episodes, ep_max_steps = 300, 200
    for ep in range(total_episodes):
        done, ep_ret, step = False, 0, 0
        state = env.reset()
        while not done and step < ep_max_steps:
            if t > start_steps: # trick for better exploration
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
