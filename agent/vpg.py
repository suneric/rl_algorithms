import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from copy import deepcopy
from .core import *

def mlp_model(input_dim, output_dim, hidden_sizes, activation, output_activation):
    """
    multiple layer perception model
    """
    input = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_sizes[0], activation=activation)(input)
    for i in range(1, len(hidden_sizes)):
        x = layers.Dense(hidden_sizes[i], activation=activation)(x)
    initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    output = layers.Dense(output_dim, activation=output_activation, kernel_initializer=initializer)(x)
    return tf.keras.Model(input, output)

class ReplayBuffer:
    """
    Replay Buffer for Policy Optimization, store experiences and calculate total rewards, advanteges
    the buffer will be used for update the policy
    """
    def __init__(self, obs_dim, act_dim, capacity, gamma=0.99, lamda=0.95):
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(capacity, dtype=np.int32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.logprob_buf = np.zeros(capacity, dtype=np.float32)
        self.gamma, self.lamda = gamma, lamda
        self.ptr, self.traj_idx = 0, 0

    def store(self, obs, act, rew, value, logprob):
        self.obs_buf[self.ptr]=obs
        self.act_buf[self.ptr]=act
        self.rew_buf[self.ptr]=rew
        self.val_buf[self.ptr]=value
        self.logprob_buf[self.ptr]=logprob
        self.ptr += 1

    def finish_trajectory(self, last_value = 0):
        """
        For each epidode, calculating the total reward and advanteges with specific
        """
        path_slice = slice(self.traj_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lamda) # GAE
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1] # rewards-to-go
        self.traj_idx = self.ptr

    def get(self):
        """
        Get all data of the buffer and normalize the advantages
        """
        self.ptr, self.traj_idx = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf-adv_mean) / adv_std
        return dict(
            obs=self.obs_buf,
            act=self.act_buf,
            adv=self.adv_buf,
            ret=self.ret_buf,
            logp=self.logprob_buf,
            )

class VPG:
    """
    Vanilla Policy Gradient
    Use the approximated value function to evaluated the policy. The trajectory performance
    $J(\pi_{\theta}) = \mathbb{E}[R(\tau)]$ could be estimated using a sample of random trajectories.
    By using D trajectories stored in memorry, the **policy gradient** is
    $\hat{g} = \frac{1}{|D|}\sum_{\tau \in D}\sum_{t=0,T}\nable_{\theta}log \pi_{\theta}(a_t|s_t)R(\tau)$
    and the $R(\tau)$ can be replaced with any Generalized Advantage Estimation, i.e.
    $\hat{A_t} = Q(s_t,a_t) - V(s_t)$
    and the policy can be updated for maximizing the objective function using gradient ascent
    $\theta_{k+1} = \theta_k + \alpha_k\hat{g_k}$
    - Facts about VPG
    1. VPG is an on-policy algorithm, it optimizes the policy using the data collected from the same policy.
    2. It can be used for environments with either discrete or continuous actions spaces
    - Limitations
    The steps taken during policy update do not guarantee an improvment in the performance and since the
    policy gradient depends on the set of trajectories collected, the complete algorithm gets trapped in
    bad loop and is not always able to recover.
    """
    def __init__(self,obs_dim,act_dim,hidden_sizes,pi_lr,q_lr,target_kl):
        self.pi = mlp_model(obs_dim,act_dim,hidden_sizes,'relu','tanh')
        self.q = mlp_model(obs_dim,1,hidden_sizes,'relu','tanh')
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.target_kl = target_kl
        self.act_dim = act_dim

    def policy(self, obs):
        state = tf.expand_dims(tf.convert_to_tensor(obs), 0)
        logits = self.pi(state)
        action = tf.squeeze(tf.random.categorical(logits,1), axis=1)
        logprob = logprobabilities(logits, action, self.act_dim)
        value = self.q(state)
        return action, logprob, value

    def learn(self, buffer, critic_iter=120):
        data = buffer.get()
        obs_buf = data['obs']
        act_buf = data['act']
        adv_buf = data['adv']
        ret_buf = data['ret']
        logprob_buf = data['logp']
        kl = 0.0
        while kl < 1.5*self.target_kl:
            kl, ent = self.update_pi(obs_buf, act_buf, logprob_buf, adv_buf)
        for _ in range(critic_iter):
            self.update_q(obs_buf, ret_buf)
        return kl, ent

    def update_pi(self, obs, act, logp, adv):
        with tf.GradientTape() as tape:
            logp_new = logprobabilities(self.pi(obs),act,self.act_dim)
            loss = -tf.reduce_mean(logp_new*adv)
        grads = tape.gradient(loss, self.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(grads, self.pi.trainable_variables))
        kl = tf.reduce_mean(logp - logprobabilities(self.pi(obs),act, self.act_dim))
        ent = tf.reduce_mean(-logprobabilities(self.pi(obs),act,self.act_dim))
        return tf.reduce_sum(kl), tf.reduce_sum(ent)

    def update_q(self, obs, ret):
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.MSE(ret, self.q(obs))
        grads = tape.gradient(loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grads, self.q.trainable_variables))
