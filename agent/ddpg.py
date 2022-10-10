import numpy as np
import tensorflow as tf
from copy import deepcopy
from .core import ActorCritic, copy_network_variables

"""
https://spinningup.openai.com/en/latest/algorithms/ddpg.html
DDPG is an algorithm which concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.
Quick Facts
- DDPG is an off-policy algorithm
- DDPG can only be used for environment with continuous actions spaces
- DDPG can be thought of as being deep Q-learning for continuous actions spaces

DDPG is an off-policy algorithm. The reason is that the Bellman equation doesn't care
which transition tuples are used, or how the actions were selected, or what happens
after a given transition, because the optimal Q-function should satisfy the Bellman
equation for all possible transition. So any transitions that we've ever experienced
are fair game when trying to fit a Q-function approximator via MSBE minimization.
"""
class DDPG:
    def __init__(self,obs_dim,act_dim,hidden_sizes,act_limit,gamma,polyak,pi_lr,q_lr, noise_obj):
        self.ac = ActorCritic(obs_dim,act_dim,hidden_sizes,'relu',act_limit)
        self.ac_target = deepcopy(self.ac)
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.gamma = gamma
        self.polyak = polyak
        self.act_limit = act_limit
        self.noise_obj = noise_obj

    def policy(self, obs):
        state = tf.expand_dims(tf.convert_to_tensor(obs),0)
        sampled_acts = tf.squeeze(self.ac.act(state)) + self.noise_obj()
        return np.clip(sampled_acts, -self.act_limit, self.act_limit)

    def learn(self, buffer):
        sampled_batch = buffer.sample()
        obs_batch = sampled_batch['obs']
        nobs_batch = sampled_batch['nobs']
        act_batch = sampled_batch['act']
        rew_batch = sampled_batch['rew']
        done_batch = sampled_batch['done']
        self.update(obs_batch, act_batch, rew_batch, nobs_batch, done_batch)

    def update(self, obs, act, rew, nobs, done):
        """
        Uses off-policy data and the Bellman equation to learn the Q-function
        Q*(s,a) = E [r(s,a) + gamma x max(Q*(s',a'))]
        minimizing MSBE loss with stochastic gradient descent
        L_p = E [(Q_p(s,a) - (r + gamma x Q_q(s', u_q(s'))))^2]
        """
        with tf.GradientTape() as tape:
            target_act = self.ac_target.pi(nobs)
            target_q = self.ac_target.q(nobs,target_act)
            q = self.ac.q(obs, act)
            y = rew + self.gamma * (1-done) * target_q
            q_loss = tf.keras.losses.MSE(y, q)
        q_grad = tape.gradient(q_loss, self.ac.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.ac.q.trainable_variables))
        """
        Use Q-function to learn policy
        Policy learning in DDPG is fairly simple. We want to learn a deterministic polict u(s) which
        gives the action that maximize Q(s,a). Because the action sapce is continuous, and we assume
        the Q-function is differentiable with respect to action, we can just perform gradient ascent
        to solve max(E [Q(s, u(s))])
        """
        with tf.GradientTape() as tape:
            q = self.ac.q(obs, self.ac.pi(obs))
            pi_loss = -tf.math.reduce_mean(q)
        pi_grad = tape.gradient(pi_loss, self.ac.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.ac.pi.trainable_variables))
        # update target network with same parameters
        copy_network_variables(self.ac_target.pi.variables, self.ac.pi.variables, self.polyak)
        copy_network_variables(self.ac_target.q.variables, self.ac.q.variables, self.polyak)
