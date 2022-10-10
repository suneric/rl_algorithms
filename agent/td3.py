import numpy as np
import tensorflow as tf
from copy import deepcopy
from .core import ActorTwinCritic, copy_network_variables

"""
https://spinningup.openai.com/en/latest/algorithms/td3.html

"""
class TD3:
    def __init__(self,obs_dim,act_dim,hidden_sizes,act_limit,gamma,polyak,pi_lr,q_lr,noise_obj):
        self.ac = ActorTwinCritic(obs_dim,act_dim,hidden_sizes,'relu',act_limit)
        self.ac_target = deepcopy(self.ac)
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.gamma = gamma
        self.polyak = polyak
        self.act_limit = act_limit
        self.noise_obj = noise_obj
        self.pi_learn_interval = 2
        self.learn_iter = 0

    def policy(self, obs):
        state = tf.expand_dims(tf.convert_to_tensor(obs),0)
        sampled_acts = tf.squeeze(self.ac.act(state)) + self.noise_obj()
        return tf.clip_by_value(sampled_acts, -self.act_limit, self.act_limit)

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
        Trick 1: learn two Q-function and use the smaller one of two Q values
        """
        with tf.GradientTape() as tape:
            """
            Trick 3: add noise to the target action, making it harder for the policy to
            exploit Q-function errors by smoothing out Q along changes in action.
            """
            target_act = self.ac_target.pi(nobs) + self.noise_obj()
            target_act = tf.clip_by_value(target_act, -self.act_limit, self.act_limit)
            target_q1, target_q2 = self.ac_target.tq(nobs, target_act)
            target_q = tf.minimum(target_q1, target_q2)
            y = rew + self.gamma * (1-done) * target_q
            q1, q2 = self.ac.tq(obs, act)
            q_loss = tf.keras.losses.MSE(y, q1) + tf.keras.losses.MSE(y, q2)
        q_grad = tape.gradient(q_loss, self.ac.tq.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.ac.tq.trainable_variables))
        """
        Trick 2: update policy and target network less frequently than Q-function
        """
        if self.learn_iter % self.pi_learn_interval == 0:
            with tf.GradientTape() as tape:
                q = self.ac.tq.Q1(obs, self.ac.pi(obs))
                pi_loss = -tf.math.reduce_mean(q)
            pi_grad = tape.gradient(pi_loss, self.ac.pi.trainable_variables)
            self.pi_optimizer.apply_gradients(zip(pi_grad, self.ac.pi.trainable_variables))
            # update target network with same parameters
            copy_network_variables(self.ac_target.pi.variables, self.ac.pi.variables, self.polyak)
            copy_network_variables(self.ac_target.tq.variables, self.ac.tq.variables, self.polyak)
