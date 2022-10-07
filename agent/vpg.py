import numpy as np
import tensorflow as tf
from copy import deepcopy
from .core import *

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
class VPG:
    def __init__(self,obs_dim,act_dim,hidden_sizes,pi_lr,q_lr,target_kl):
        self.actor = mlp_model(obs_dim,act_dim,hidden_sizes,'relu',None)
        self.critic = mlp_model(obs_dim,1,hidden_sizes,'relu',None)
        print(self.actor.summary())
        print(self.critic.summary())
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.target_kl = target_kl
        self.act_dim = act_dim

    def policy(self, obs):
        state = tf.expand_dims(tf.convert_to_tensor(obs), 0)
        logits = self.actor(state)
        action = tf.squeeze(tf.random.categorical(logits,1), axis=1)
        logprob = self.logprobabilities(logits, action)
        value = self.critic(state)
        return action, logprob, value

    def logprobabilities(self, logits, action):
        """
        Compute the log-probabilities of taking actions by using logits (e.g. output of the actor)
        """
        logprob_all = tf.nn.log_softmax(logits)
        logprob = tf.reduce_sum(tf.one_hot(action, self.act_dim)*logprob_all, axis=1)
        return logprob

    def learn(self, buffer, critic_iter=80):
        data = buffer.get()
        obs_buf = data['obs']
        act_buf = data['act']
        adv_buf = data['adv']
        ret_buf = data['ret']
        logprob_buf = data['logp']
        kl, ent = self.update_pi(obs_buf, act_buf, logprob_buf, adv_buf)
        for _ in range(critic_iter):
            self.update_q(obs_buf, ret_buf)
        return kl, ent

    def update_pi(self, obs, act, logp, adv):
        with tf.GradientTape() as tape:
            logp_new = self.logprobabilities(self.actor(obs),act)
            loss = -tf.reduce_mean(logp_new*adv)
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        kl = tf.reduce_mean(logp - self.logprobabilities(self.actor(obs),act))
        ent = tf.reduce_mean(-self.logprobabilities(self.actor(obs),act))
        return tf.reduce_sum(kl), tf.reduce_sum(ent)

    def update_q(self, obs, ret):
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.MSE(ret, self.critic(obs))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
