import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from copy import deepcopy
from .core import GaussianActor, TwinCritic, copy_network_variables

"""
https://spinningup.openai.com/en/latest/algorithms/sac.html#

"""
class SAC:
    def __init__(self,obs_dim,act_dim,hidden_sizes,act_limit,gamma,polyak,pi_lr,q_lr,temp):
        self.pi = GaussianActor(obs_dim,act_dim,hidden_sizes,'relu',act_limit)
        self.tq = TwinCritic(obs_dim,act_dim,hidden_sizes,'relu')
        self.tq_target = deepcopy(self.tq)
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.gamma = gamma
        self.polyak = polyak
        self.act_limit = act_limit
        self.temperature = temp
        self.pi_learn_interval = 2
        self.learn_iter = 0

    def policy(self, obs):
        state = tf.expand_dims(tf.convert_to_tensor(obs),0)
        action, logprob = self.pi(state)
        return tf.clip_by_value(tf.squeeze(action), -self.act_limit, self.act_limit)

    def learn(self, buffer):
        sampled_batch = buffer.sample()
        obs_batch = sampled_batch['obs']
        nobs_batch = sampled_batch['nobs']
        act_batch = sampled_batch['act']
        rew_batch = sampled_batch['rew']
        done_batch = sampled_batch['done']
        self.update(obs_batch, act_batch, rew_batch, nobs_batch, done_batch)

    def update(self, obs, act, rew, nobs, done):
        self.learn_iter += 1
        """
        Like TD3, learn two Q-function and use the smaller one of two Q values
        """
        with tf.GradientTape() as tape:
            """
            Unlike TD3, use current policy to get next action
            """
            nact, nact_logp = self.pi(nobs)
            target_q1, target_q2 = self.tq_target(nobs, nact)
            target_q = tf.minimum(target_q1, target_q2) - self.temperature * nact_logp
            y = rew + self.gamma * (1-done) * target_q
            q1, q2 = self.tq(obs, act)
            q_loss = tf.keras.losses.MSE(y, q1) + tf.keras.losses.MSE(y, q2)
        q_grad = tape.gradient(q_loss, self.tq.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.tq.trainable_variables))
        """
        update policy and target network less frequently than Q-function
        """
        if self.learn_iter % self.pi_learn_interval == 0:
            with tf.GradientTape() as tape:
                action, action_logp = self.pi(obs)
                q1, q2 = self.tq(obs, action)
                adv = tf.stop_gradient(action_logp - tf.minimum(q1,q2))
                pi_loss = -tf.math.reduce_mean(action_logp*adv)
            pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
            self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
            # update target Q network with same parameters
            copy_network_variables(self.tq_target.variables, self.tq.variables, self.polyak)
