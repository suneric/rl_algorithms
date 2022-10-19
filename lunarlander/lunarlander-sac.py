import sys
sys.path.append('..')
sys.path.append('.')
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
from agent.sac import SAC, ReplayBuffer

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
    logDir = 'logs/sac' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    env = gym.make("LunarLander-v2", continuous=True, render_mode='human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    print("state {}, action {}, limit {}".format(obs_dim,act_dim,act_limit))

    buffer = ReplayBuffer(obs_dim,act_dim,capacity=50000,batch_size=128)
    agent = SAC(obs_dim,act_dim,hidden_sizes=[400,400],
        act_limit=act_limit,gamma=0.99,polyak=0.995,pi_lr=2e-4,q_lr=3e-4,alpha_lr=2e-4,alpha=0.2,auto_ent=True)

    ep_ret_list, avg_ret_list = [], []
    t, start_steps, update_after, update_freq = 0, 1e4, 1e3, 50
    total_episodes, ep_max_steps = 500, 1000
    for ep in range(total_episodes):
        done, ep_ret, step = False, 0, 0
        state = env.reset()
        while not done and step < ep_max_steps:
            if t > start_steps:  # trick for better exploration
                a = agent.policy(state[0])
            else:
                a = env.action_space.sample()
            new_state = env.step(a)
            r, done = new_state[1], new_state[2]
            buffer.store(state[0],a,r,new_state[0],done)
            t += 1
            step += 1
            ep_ret += r
            state = new_state

            if t > update_after and t % update_freq == 0:
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
