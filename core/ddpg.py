"""
https://spinningup.openai.com/en/latest/algorithms/ddpg.html
DDPG is an algorithm which concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.
Quick Facts
- DDPG is an off-policy algorithm
- DDPG can only be used for environment with continuous actions spaces
- DDPG can be thought of as being deep Q-learning for continuous actions spaces

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from copy import deepcopy
from .core import ActorCritic

"""
Replay Buffer for storing experiences
All standard algorithm for training a DNN to approximator Q*(s,a) make use of an experience replay buffer.
This is the set D of previous experiences. In order for the algorithm to have stable behavior, the replay
buffer should be large enough to contain a wide range of experiences, but it may not always be good to keep
everything. If you only use the very-most recent data, you will overfit to that and things will break; if
you use too much experience, you may slow down your learning. This may take some tuning to get right.
"""
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, batch_size):
        self.obs_buf = np.zeros((size, obs_dim),dtype=np.float32)
        self.nobs_buf = np.zeros((size, obs_dim),dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim),dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.batch_size = batch_size

    """
    Takes (s,a,r,s') observation tuple as input
    """
    def store(self, obs, act, rew, nobs, done):
        self.obs_buf[self.ptr] = obs
        self.nobs_buf[self.ptr] = nobs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    """
    Sampling
    """
    def sample(self):
        idxs = np.random.randint(0,self.size,size=self.batch_size)
        return dict(
            obs = tf.convert_to_tensor(self.obs_buf[idxs]),
            nobs = tf.convert_to_tensor(self.nobs_buf[idxs]),
            act = tf.convert_to_tensor(self.act_buf[idxs]),
            rew = tf.convert_to_tensor(self.rew_buf[idxs]),
            done = tf.convert_to_tensor(self.done_buf[idxs])
        )


"""
DDPG is an off-policy algorithm. The reason is that the Bellman equation doesn't care
which transition tuples are used, or how the actions were selected, or what happens
after a given transition, because the optimal Q-function should satisfy the Bellman
equation for all possible transition. So any transitions that we've ever experienced
are fair game when trying to fit a Q-function approximator via MSBE minimization.
"""
class DDPG:
    def __init__(self,obs_dim,act_dim,hidden_sizes,act_limit,gamma,polyak,pi_lr,q_lr):
        self.ac = ActorCritic(obs_dim,act_dim,hidden_sizes,'relu',act_limit)
        self.ac_target = deepcopy(self.ac)
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.gamma = gamma
        self.polyak = polyak
        self.act_limit = act_limit

    def policy(self, obs, noise):
        state = tf.expand_dims(tf.convert_to_tensor(obs),0)
        sampled_acts = tf.squeeze(self.ac.act(state)) + noise
        return np.clip(sampled_acts, -self.act_limit, self.act_limit)

    def learn(self, buffer):
        sampled_batch = buffer.sample()
        obs_batch = sampled_batch['obs']
        nobs_batch = sampled_batch['nobs']
        act_batch = sampled_batch['act']
        rew_batch = sampled_batch['rew']
        done_batch = sampled_batch['done']
        self.update_policy(obs_batch, act_batch, rew_batch, nobs_batch, done_batch)
        self.update_target(self.ac_target.pi.variables, self.ac.pi.variables)
        self.update_target(self.ac_target.q.variables, self.ac.q.variables)

    @tf.function
    def update_policy(self, obs, act, rew, nobs, done):
        """
        Uses off-policy data and the Bellman equation to learn the Q-function
        Q*(s,a) = E [r(s,a) + gamma x max(Q*(s',a'))]
        minimizing MSBE loss with stochastic gradient descent
        L_p = E [(Q_p(s,a) - (r + gamma x Q_q(s', u_q(s'))))^2]
        """
        with tf.GradientTape() as tape:
            q = self.ac.q([obs, act])
            target_q = self.ac_target.q([nobs,self.ac_target.pi(nobs)])
            y = rew + self.gamma * (1-done) * target_q
            q_loss = tf.math.reduce_mean(tf.math.square(y - q))
        grad = tape.gradient(q_loss, self.ac.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grad, self.ac.q.trainable_variables))

        """
        Use Q-function to learn policy
        Policy learning in DDPG is fairly simple. We want to learn a deterministic polict u(s) which
        gives the action that maximize Q(s,a). Because the action sapce is continuous, and we assume
        the Q-function is differentiable with respect to action, we can just perform gradient ascent
        to solve max(E [Q(s, u(s))])
        """
        with tf.GradientTape() as tape:
            q = self.ac.q([obs, self.ac.pi(obs)])
            pi_loss = -tf.math.reduce_mean(q) # '-' for gradient ascent
        grad = tape.gradient(pi_loss, self.ac.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(grad, self.ac.pi.trainable_variables))

    @tf.function
    def update_target(self,target_weights, weights):
        """
        In DQN-based algorithms, the target network is just copied over from the main network
        every some-fixed-number of steps. In DDPG-style algorithm, the target network is updated
        once per main network update by polyak averaging, where polyak(tau) usually close to 1.
        """
        for (a,b) in zip(target_weights, weights):
            a.assign(a*self.polyak + b*(1-self.polyak))


"""
Gaussian Noise added to Action for better exploration
DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the
agent were to explore on-policy, int the beginning it would probably not try a wide ennough varienty
of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their
actions at traiing time. Uncorreletaed, mean-zero Gaussian noise work perfectly well, and it is suggested
as it is simpler. At test time, to see how well the policy exploits what it has learned, we don not add
noise to the actions.
"""
class GSNoise:
    def __init__(self, mean, std_dev, size):
        self.mean = mean
        self.std_dev = std_dev
        self.size = size

    def __call__(self):
        return np.random.normal(self.mean,self.std_dev,self.size)

"""
Ornstein Uhlenbeck process
"""
class OUNoise:
    def __init__(self,mean,std_dev,theta=0.15,dt=1e-2,x_init=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.x_init = x_init
        self.dt = dt
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_init is not None:
            self.x_prev = self.x_init
        else:
            self.x_prev = np.zeros_like(self.mean)
