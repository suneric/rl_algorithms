import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from copy import deepcopy
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
    def __init__(self, obs_dim, act_dim, hidden_sizes, pi_lr, q_lr, target_kld):
        self.pi = mlp_net(obs_dim,act_dim,hidden_sizes,'relu','linear')
        self.q = mlp_net(obs_dim,1,hidden_sizes,'relu','linear')
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.act_dim = act_dim
        self.target_kld = target_kld

    def policy(self, obs):
        """
        return action given an observation based on policy network
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

    def learn(self, buffer, iter=80):
        experiences = buffer.get()
        obs_batch = experiences['obs']
        act_batch = experiences['act']
        ret_batch = experiences['ret']
        adv_batch = experiences['adv']
        logp_batch = experiences['logp']
        self.update(obs_batch, act_batch, ret_batch, adv_batch, logp_batch, iter)

    def update(self, obs, act, ret, adv, old_logp, iter):
        """
        In REINFORCE (VPG) as well as other policy gradient algorithms, the gradient steps taken
        aim to minimize a loss function by incrementally modifying the policy network's parameters.
        The loss function in original VPG aglorithm is given by:
            -log(pi(s,a))*G
        where the 'log' term is the log probability of taking some action 'a' at some state 's',
        and 'G' is the return, the sum of the discounted reward from the current timestep up until
        the end of the episode. In practice, this loss function isn't typically used as its performance
        is limited by the high variance in the return G over entire episodes. To combat this, an
        advantage estimate is introduced in place of the return G. This advantage is given by:
            A(s,a) = r + gamma*V(s')-V(s)
        where 'V(s)' is a learned value function that estimates the value of given state, 'r' is the
        reward received from transitioning from state 's' into state 's'' by taking action 'a', 'gamma'
        is the discount rate, a hyperparameter passed to the algorithm.
        The augmented loss function then bacomes:
            -log(pi(s,a))*A(s,a)
        Naturally, since the value function is learned over time as more updates are performed, it
        introduces some margin bias caused by the imperfect estimates, but decrease the overalll
        variance as well.
        """
        with tf.GradientTape() as tape:
            tape.watch(self.pi.trainable_variables)
            logits = self.pi(obs)
            logp = tfp.distributions.Categorical(logits=logits).log_prob(act)
            approx_kld = old_logp-logp
            pi_loss = -tf.reduce_mean(logp*adv)
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        for _ in range(iter):
            self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
            if tf.math.reduce_mean(approx_kld) > self.target_kld:
                break
        """
        Fit value network
        """
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            pred_q = self.q(obs)
            q_loss = tf.keras.losses.MSE(ret, pred_q)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        for _ in range(iter):
            self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))
