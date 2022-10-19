import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from .core import *

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, gamma=0.99, lamda=0.95):
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(capacity, dtype=np.int32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.gamma, self.lamda = gamma, lamda
        self.ptr, self.traj_idx = 0, 0

    def store(self, obs, act, rew, val, logp):
        self.obs_buf[self.ptr]=obs
        self.act_buf[self.ptr]=act
        self.rew_buf[self.ptr]=rew
        self.val_buf[self.ptr]=val
        self.logp_buf[self.ptr]=logp
        self.ptr += 1

    def finish_trajectory(self, last_value = 0):
        """
        For each epidode, calculating the total reward and advanteges
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
        s = slice(0,self.ptr)
        advs = self.adv_buf[s]
        normalized_advs = (advs-np.mean(advs)) / (np.std(advs)+1e-10)
        data = dict(
            obs=self.obs_buf[s],
            act=self.act_buf[s],
            ret=self.ret_buf[s],
            logp=self.logp_buf[s],
            adv=normalized_advs,
            )
        self.ptr, self.traj_idx = 0, 0
        return data

class PPO:
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
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, pi_lr, q_lr, clip_ratio, beta, target_kld):
        self.pi = mlp_net(obs_dim,act_dim,hidden_sizes,'relu','linear')
        self.q = mlp_net(obs_dim,1,hidden_sizes,'relu','linear')
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.act_dim = act_dim
        self.clip_r = clip_ratio
        self.beta = beta
        self.target_kld = target_kld

    def policy(self, obs):
        """
        return action and log probability given an observation based on policy network
        """
        logits = self.pi(tf.expand_dims(tf.convert_to_tensor(obs),0))
        dist = tfp.distributions.Categorical(logits=logits)
        action = tf.squeeze(dist.sample()).numpy()
        logprob = tf.squeeze(dist.log_prob(action)).numpy()
        return action, logprob

    def value(self, obs):
        """
        return q value of an observation based on q-function network
        """
        val = self.q(tf.expand_dims(tf.convert_to_tensor(obs),0))
        return tf.squeeze(val).numpy()

    def learn(self, buffer, pi_iter=80, q_iter=80):
        experiences = buffer.get()
        obs_batch = experiences['obs']
        act_batch = experiences['act']
        ret_batch = experiences['ret']
        adv_batch = experiences['adv']
        logp_batch = experiences['logp']
        self.update(obs_batch, act_batch, ret_batch, adv_batch, logp_batch, pi_iter, q_iter)

    def update(self, obs, act, ret, adv, old_logp, pi_iter, q_iter):
        """
        The objective function of TRPO is J = E[ratio*A(s,a)], where ratio is probability
        ratio between old and new policies as ratio = pi(a|s)/pi_old(a|s)
        without a limitation on the distance between old policy and new policy, to maximize J would
        lead to instability with extremely large parameter updates and big policy ratios.
        PPO imposes the constrain by forcing 'ratio' to stay with a small intervel around 1, [1-e, 1+e]
        where e is a hyperparameter, usually is 0.2. Thus, the objective function of PPO takes
        the minimum one bewteen original value and the clipped version and therefore we lose the
        motivation for increasing the policy update to extremes for better rewards.
        When applying PPO on the network architecture with shared parameters for both policy and value
        functions, in addition to the clipped reward, the objective function is augmented with an error
        term on the value estimation and an entropy term to encourage sufficient exploration.
            J' = E[J - c1*(V(s)-V_target)^2 +c2*H(s,pi(.))]
        """
        with tf.GradientTape() as tape:
            logits=self.pi(obs,training=True)
            logp = tfp.distributions.Categorical(logits=logits).log_prob(act)
            ratio = tf.exp(logp - old_logp) # pi/old_pi
            clip_adv = tf.clip_by_value(ratio, 1-self.clip_r, 1+self.clip_r)*adv
            approx_kld = old_logp-logp
            pmf = tf.nn.softmax(logits=logits) # probability
            ent = tf.math.reduce_sum(-pmf*tf.math.log(pmf),axis=-1) # entropy
            pi_loss = -tf.math.reduce_mean(tf.math.minimum(ratio*adv, clip_adv)) + self.beta*ent
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        for _ in range(pi_iter):
            self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
            if tf.math.reduce_mean(approx_kld) > self.target_kld:
                break
        """
        Fit value network
        """
        with tf.GradientTape() as tape:
            val = self.q(obs, training=True)
            q_loss = tf.keras.losses.MSE(ret, val)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        for _ in range(q_iter):
            self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))
