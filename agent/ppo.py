import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from .core import *

"""
PPO with tensorflow implementation

The goal of RL is to find an optimal behavior strategy for the agent to obtain
optimal rewards. The policy gradient methods target at modeling and optimizing
the policy directly. The policy loss is defined as
    L = E [log pi (a|s)] * AF
where, 'L' is the policy loss, 'E' is the expected, 'log pi(a|s)' log probability
of taking the action at that state. 'AF' is the advantage.

PPO is an on-policy algorithm which can be used for environments with either discrete
or continous actions spaces. There are two primary variants of PPO: PPO-penalty which
approximately solves a KL-constrained update like TRPO, but penalizes the KL-divergence
in the objective function instead of make it a hard constraint; PPO-clip which does not
have a KL-divergence term in the objective and does not have a constraint at all,
instead relies on specialized clipping in the objective function to remove incentives
for the new policy to get far from the old policy. This implementation uses PPO-clip.

PPO is a policy gradient method and can be used for environments with either discrete
or continuous action spaces. It trains a stochastic policy in an on-policy way. Also,
it utilizes the actor critic method. The actor maps the observation to an action and
the critic gives an expectation of the rewards of the agent for the observation given.
Firstly, it collects a set of trajectories for each epoch by sampling from the latest
version of the stochastic policy. Then, the rewards-to-go and the advantage estimates
are computed in order to update the policy and fit the value function. The policy is
updated via a stochastic gradient ascent optimizer, while the value function is fitted
via some gradient descent algorithm. This procedure is applied for many epochs until
the environment is solved.

references:
[1] https://arxiv.org/pdf/1707.06347.pdf
[2] https://spinningup.openai.com/en/latest/algorithms/ppo.html
[3] https://github.com/suneric/door_open_rl/blob/main/do_learning/scripts/agents/ppo_mixed.py
"""
class PPO:
    def __init__(self, obs_dim, act_dim, hidden_sizes, clip_ratio, actor_lr, critic_lr, target_kl):
        self.actor = mlp_model(obs_dim, act_dim, hidden_sizes, 'relu', None)
        self.critic = mlp_model(obs_dim,1,hidden_sizes,'relu', None)
        print(self.actor.summary())
        print(self.critic.summary())
        self.pi_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.clip_r = clip_ratio
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

    def learn(self, buffer, actor_iter=80, critic_iter=80):
        data = buffer.get()
        obs_buf = data['obs']
        act_buf = data['act']
        adv_buf = data['adv']
        ret_buf = data['ret']
        logprob_buf = data['logp']
        for _ in range(actor_iter):
            kl = self.update_pi(obs_buf, act_buf, logprob_buf, adv_buf)
            if kl > 1.5 * self.target_kl:
                break # Early Stopping
        for _ in range(critic_iter):
            self.update_q(obs_buf, ret_buf)

    def update_pi(self, obs, act, logp, adv):
        with tf.GradientTape() as tape:
            ratio = tf.exp(self.logprobabilities(self.actor(obs),act) - logp)
            min_adv = tf.where(adv > 0, (1+self.clip_r)*adv, (1-self.clip_r)*adv)
            loss = -tf.reduce_mean(tf.minimum(ratio*adv, min_adv))
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        kl = tf.reduce_mean(logp - self.logprobabilities(self.actor(obs), act))
        return tf.reduce_sum(kl)

    def update_q(self, obs, ret):
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.MSE(ret, self.critic(obs))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
