import sys
sys.path.append('..')
sys.path.append('.')
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
from agent.ddpg import DDPG
from agent.core import ReplayBuffer, OUNoise, GSNoise

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
Limiting GPU memory growth
"""
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

np.random.seed(123)

if __name__ == '__main__':
    env = gym.make("Pendulum-v1", render_mode = 'human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    print("state {}, action {}, limit {}".format(obs_dim,act_dim,act_limit))

    buffer = ReplayBuffer(obs_dim,act_dim,capacity=500000,batch_size=64)
    noise = GSNoise(mean=0,std_dev=0.2*act_limit,size=act_dim)
    #noise = OUNoise(x=np.zeros(act_dim), mean=0,std_dev=0.1,theta=0.15,dt=0.01)
    hidden_sizes = [64,64]
    gamma = 0.99
    polyak = 0.995
    pi_lr = 1e-4
    q_lr = 2e-4
    agent = DDPG(obs_dim,act_dim,hidden_sizes,act_limit,gamma,polyak,pi_lr,q_lr)

    logDir = 'logs/ddpg' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    total_episodes = 1000
    t, start_steps, update_after, ep_max_step = 0, 5000, 1000, 200
    ep_ret_list, avg_ret_list = [], []
    for ep in range(total_episodes):
        ep_ret, ep_step = 0, 0
        done = False
        state = env.reset()
        o = state[0]
        while not done and ep_step < ep_max_step:
            if t > start_steps:
                a = agent.policy(o, noise())
            else: # randomly select sample actions for better exploration
                a = env.action_space.sample()
            state = env.step(a)
            o2,r,done = state[0],state[1],state[2]
            buffer.store(o,a,r,o2,done)
            t += 1
            ep_step += 1
            ep_ret += r
            o = o2

            if t > update_after:
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
