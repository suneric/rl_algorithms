import sys
sys.path.append('..')
sys.path.append('.')
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
from core.ddpg import *

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
Limiting GPU memory growth
"""
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

np.random.seed(123)

if __name__ == '__main__':
    env = gym.make("Pendulum-v1", render_mode='human')
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    print("state {}, action {}, limit {}".format(obs_dim,act_dim,act_limit))

    buffer = ReplayBuffer(obs_dim,act_dim,size=50000,batch_size=128)
    noise = GSNoise(mean=0,std_dev=0.1*act_limit,size=act_dim)
    # noise = OUActionNoise(mean=np.zeros(act_dim),std_dev=0.1*act_limit)
    hidden_sizes = [64,64]
    gamma = 0.99
    polyak = 0.995
    pi_lr = 1e-3
    q_lr=1e-3
    agent = DDPG(obs_dim,act_dim,hidden_sizes,act_limit,gamma,polyak,pi_lr,q_lr)

    logDir = 'logs/ddpg' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    start_steps = 5000
    update_after = 1000
    update_every = 50

    total_episodes = 1000
    max_steps = 200

    t = 0
    ep_ret_list, avg_ret_list = [], []
    for ep in range(total_episodes):
        state = env.reset()
        o = state[0]
        ep_ret = 0
        for _ in range(max_steps):
            if t > start_steps:
                a = agent.policy(o, noise())
            else: # randomly select sample actions for better exploration
                a = env.action_space.sample()

            state = env.step(a)
            o2,r,d = state[0],state[1],state[2]
            buffer.store(o,a,r,o2,d)

            o = o2
            ep_ret += r
            t += 1

            # update the network
            if t > update_after and t % update_every == 0:
                for _ in range(update_every):
                    agent.learn(buffer)

            if d:
                break

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
