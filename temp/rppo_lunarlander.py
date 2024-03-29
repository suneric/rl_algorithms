import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import deque
import tensorflow_probability as tfp
import scipy.signal
from collections import deque

np.random.seed(123)
tf.random.set_seed(123)

def discount_cumsum(x,discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def logprobabilities(logits, action, action_dim):
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(action, action_dim) * logprobabilities_all, axis=1
    )
    return logprobability

def smoothExponential(data, weight):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def copy_network_variables(target_weights, from_weights, polyak = 0.0):
    for (a,b) in zip(target_weights, from_weights):
        a.assign(a*polyak + b*(1-polyak))

def build_mlp_model(n_hidden,n_input,n_output,activation='tanh',output_activation=None):
    input = layers.Input(shape=(n_input,))
    x = layers.Dense(n_hidden, activation=activation)(input)
    x = layers.Dense(n_hidden, activation=activation)(x)
    output = layers.Dense(n_output,activation=output_activation)(x)
    model = keras.Model(input,output,name='mlp')
    print(model.summary())
    return model

def build_rnn_model(n_hidden,n_input,n_output,seq_len,activation='tanh',output_activation=None):
    input = layers.Input(shape=(seq_len,n_input)) # given a fixed seqence length
    x = layers.LSTM(n_hidden,activation=activation,return_sequences=False)(input)
    x = layers.Dense(n_hidden, activation=activation)(x)
    output = layers.Dense(n_output,activation=output_activation)(x)
    model = keras.Model(input,output,name='rnn')
    print(model.summary())
    return model

def zero_obs_seq(obs_dim,seq_len):
    obs_seq = deque(maxlen=seq_len)
    for _ in range(seq_len):
        obs_seq.append(np.zeros(obs_dim))
    return obs_seq

class PPORolloutBuffer:
    def __init__(self, obs_dim, capacity, gamma=0.99, lamda=0.95, seq_len=None):
        self.obs_dim = obs_dim
        self.capacity = capacity
        self.gamma, self.lamda = gamma, lamda
        self.seq_len = seq_len
        self.recurrent = self.seq_len is not None
        self.ptr, self.traj_idx = 0, 0
        self.reset()

    def reset(self):
        self.obs_buf = np.zeros((self.capacity,self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self.capacity, dtype=np.int32)
        self.rew_buf = np.zeros(self.capacity, dtype=np.float32)
        self.ret_buf = np.zeros(self.capacity, dtype=np.float32)
        self.val_buf = np.zeros(self.capacity, dtype=np.float32)
        self.adv_buf = np.zeros(self.capacity, dtype=np.float32)
        self.logp_buf = np.zeros(self.capacity, dtype=np.float32)
        if self.recurrent:
            self.obs_seq_buf = np.zeros((self.capacity,self.seq_len,self.obs_dim), dtype=np.float32)

    def add_sample(self, obs, act, rew, val, logp):
        self.obs_buf[self.ptr]=obs
        self.act_buf[self.ptr]=act
        self.rew_buf[self.ptr]=rew
        self.val_buf[self.ptr]=val
        self.logp_buf[self.ptr]=logp
        self.ptr += 1

    def end_trajectory(self, last_value = 0):
        path_slice = slice(self.traj_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lamda) # GAE
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1] # rewards-to-go
        if self.recurrent:
            obs_seq = zero_obs_seq(self.obs_dim, self.seq_len)
            for i in range(self.traj_idx, self.ptr):
                obs_seq.append(self.obs_buf[i])
                self.obs_seq_buf[i] = np.array(obs_seq.copy())
        self.traj_idx = self.ptr

    def sample(self):
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        batch = (
            self.obs_seq_buf if self.recurrent else self.obs_buf,
            self.act_buf,
            self.ret_buf,
            self.logp_buf,
            self.adv_buf,
        )
        self.ptr, self.traj_idx = 0, 0
        self.reset()
        return batch

class PPOAgent:
    def __init__(self, actor, critic, pi_lr, q_lr, clip_ratio, target_kld, beta):
        self.pi = actor
        self.q = critic
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.clip_r = clip_ratio
        self.target_kld = target_kld
        self.beta = beta

    def policy(self, obs):
        obs = tf.expand_dims(tf.convert_to_tensor(obs),0)
        pmf = tfp.distributions.Categorical(logits=self.pi(obs))
        action = tf.squeeze(pmf.sample()).numpy()
        logprob = tf.squeeze(pmf.log_prob(action)).numpy()
        return action, logprob

    def value(self, obs):
        obs = tf.expand_dims(tf.convert_to_tensor(obs),0)
        val = self.q(obs)
        return tf.squeeze(val).numpy()

    def update_policy(self,obs,act,old_logp,adv):
        with tf.GradientTape() as tape:
            tape.watch(self.pi.trainable_variables)
            pmf = tfp.distributions.Categorical(logits=self.pi(obs))
            logp = pmf.log_prob(act)
            ratio = tf.exp(logp-old_logp) # pi/old_pi
            clip_adv = tf.clip_by_value(ratio, 1-self.clip_r, 1+self.clip_r)*adv
            obj = tf.minimum(ratio*adv, clip_adv) + self.beta*pmf.entropy()
            pi_loss = -tf.reduce_mean(obj)
            approx_kld = old_logp - logp
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
        kld = tf.reduce_mean(approx_kld)
        return kld

    def update_value_function(self,obs,ret):
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            val = self.q(obs)
            q_loss = tf.reduce_mean((ret-val)**2)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))

    def learn(self,buffer,pi_iter=80,q_iter=80,batch_size=64):
        (obs_buf,act_buf,ret_buf,logp_buf,adv_buf) = buffer.sample()
        for _ in range(pi_iter):
            idxs = np.random.choice(buffer.capacity,batch_size)
            obs = tf.convert_to_tensor(obs_buf[idxs])
            act = tf.convert_to_tensor(act_buf[idxs])
            logp = tf.convert_to_tensor(logp_buf[idxs])
            adv = tf.convert_to_tensor(adv_buf[idxs])
            kld = self.update_policy(obs,act,logp,adv)
            # print(kld)
            # if kld > self.target_kld:
            #     break
        for _ in range(q_iter):
            idxs = np.random.choice(buffer.capacity,batch_size)
            obs = tf.convert_to_tensor(obs_buf[idxs])
            ret = tf.convert_to_tensor(ret_buf[idxs])
            self.update_value_function(obs,ret)

def rppo_train(env, num_episodes, train_steps,max_step):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("state {}, action {}".format(obs_dim, act_dim))
    seq_len = 64
    actor = build_rnn_model(n_hidden=64,n_input=obs_dim,n_output=act_dim,seq_len=seq_len)
    critic = build_rnn_model(n_hidden=64,n_input=obs_dim,n_output=1,seq_len=seq_len)
    buffer = PPORolloutBuffer(obs_dim,capacity=train_steps,gamma=0.99,lamda=0.97,seq_len=seq_len)
    agent = PPOAgent(actor,critic,pi_lr=3e-4,q_lr=1e-3,clip_ratio=0.2,target_kld=1e-2, beta=1e-3)

    ep_returns, t = [], 0
    for ep in range(num_episodes):
        state, done, ep_ret, step = env.reset(), False, 0, 0
        o_seq = zero_obs_seq(obs_dim,seq_len)
        while step < max_step:
            o = state[0]
            o_seq.append(o)
            a, logp = agent.policy(o_seq)
            value = agent.value(o_seq)
            new_state = env.step(a)
            r, done = new_state[1], new_state[2]
            buffer.add_sample(o,a,r,value,logp)
            state = new_state
            ep_ret += r
            t += 1
            step += 1
            if done or (t % train_steps == 0):
                o1_seq = o_seq.copy()
                o1_seq.append(state[0])
                last_value = 0 if done else agent.value(o1_seq)
                buffer.end_trajectory(last_value)
                break

        if t % train_steps == 0:
            agent.learn(buffer)

        ep_returns.append(ep_ret)
        print("Episode: {}, Total Return: {:.4f}, Total Steps: {}".format(ep+1, ep_ret, t))
    return ep_returns

def ppo_train(env, num_episodes, train_steps, max_step):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("state {}, action {}".format(obs_dim, act_dim))
    actor = build_mlp_model(n_hidden=128,n_input=obs_dim,n_output=act_dim)
    critic = build_mlp_model(n_hidden=128,n_input=obs_dim,n_output=1)
    buffer = PPORolloutBuffer(obs_dim,capacity=train_steps,gamma=0.99,lamda=0.97)
    agent = PPOAgent(actor,critic,pi_lr=3e-4,q_lr=1e-3,clip_ratio=0.2,target_kld=1e-2,beta=1e-3)

    ep_returns, t = [], 0
    for ep in range(num_episodes):
        state, done, ep_ret, step = env.reset(), False, 0, 0
        while step < max_step:
            o = state[0]
            a, logp = agent.policy(o)
            value = agent.value(o)
            new_state = env.step(a)
            r, done = new_state[1], new_state[2]
            buffer.add_sample(o,a,r,value,logp)
            state = new_state
            ep_ret += r
            t += 1
            step += 1
            if done or (t % train_steps == 0):
                last_value = 0 if done else agent.value(state[0])
                buffer.end_trajectory(last_value)
                break

        if t % train_steps == 0:
            agent.learn(buffer)

        ep_returns.append(ep_ret)
        print("Episode: {}, Total Return: {:.4f}, Total Steps: {}".format(ep+1, ep_ret, t))
    return ep_returns

if __name__ == '__main__':
    env = gym.make("LunarLander-v2", continuous=False, render_mode='human')
    returns = rppo_train(env,500,1500,500)
    env.close()
    plt.plot(returns,'k--',linewidth=1)
    plt.plot(smoothExponential(returns,0.95),'g-',linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend(["Return", "Smoothed Return"])
    plt.show()
