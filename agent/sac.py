import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from copy import deepcopy
from .core import *

def gaussian_actor_model(obs_dim, act_dim, hidden_sizes, activation):
    last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    input = layers.Input(shape=(obs_dim,))
    x = layers.Dense(hidden_sizes[0], activation=activation)(input)
    for i in range(1, len(hidden_sizes)):
        x = layers.Dense(hidden_sizes[i], activation=activation)(x)
    output_mu = layers.Dense(act_dim, kernel_initializer=last_init)(x)
    output_logstd = layers.Dense(act_dim, kernel_initializer=last_init)(x)
    model = tf.keras.Model(input, [output_mu, output_logstd])
    return model

def twin_critic_model(obs_dim, act_dim, hidden_sizes, activation):
    obs_input = layers.Input(shape=(obs_dim))
    act_input = layers.Input(shape=(act_dim))
    x0 = layers.Concatenate()([obs_input, act_input])
    x1 = layers.Dense(hidden_sizes[0], activation=activation)(x0)
    for i in range(1, len(hidden_sizes)):
        x1 = layers.Dense(hidden_sizes[i], activation=activation)(x1)
    output_1 = layers.Dense(1)(x1)
    x2 = layers.Dense(hidden_sizes[0], activation=activation)(x0)
    for i in range(1, len(hidden_sizes)):
        x2 = layers.Dense(hidden_sizes[i], activation=activation)(x2)
    output_2 = layers.Dense(1)(x2)
    model = tf.keras.Model([obs_input, act_input], [output_1, output_2])
    return model

class ReplayBuffer:
    """
    Replay Buffer for Q-learning
    All standard algorithm for training a DNN to approximator Q*(s,a) make use of an experience replay buffer.
    This is the set D of previous experiences. In order for the algorithm to have stable behavior, the replay
    buffer should be large enough to contain a wide range of experiences, but it may not always be good to keep
    everything. If you only use the very-most recent data, you will overfit to that and things will break; if
    you use too much experience, you may slow down your learning. This may take some tuning to get right.
    """
    def __init__(self, obs_dim, act_dim, capacity, batch_size):
        self.obs_buf = np.zeros((capacity, obs_dim),dtype=np.float32)
        self.nobs_buf = np.zeros((capacity, obs_dim),dtype=np.float32)
        self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, capacity
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
        #idxs = np.random.randint(0,self.size,size=self.batch_size)
        # choice is faster than randint
        idxs = np.random.choice(self.size, self.batch_size)
        return dict(
            obs = tf.convert_to_tensor(self.obs_buf[idxs]),
            nobs = tf.convert_to_tensor(self.nobs_buf[idxs]),
            act = tf.convert_to_tensor(self.act_buf[idxs]),
            rew = tf.convert_to_tensor(self.rew_buf[idxs]),
            done = tf.convert_to_tensor(self.done_buf[idxs])
        )

class SAC:
    """
    Soft Actor Critic (SAC) is an algorithm that optimizes a stochastic policy in an off-policy way,
    forming a bridge between stochastic policy optimization and DDPG-style approaches. It isn’t a
    direct successor to TD3 (having been published roughly concurrently), but it incorporates the
    clipped double-Q trick, and due to the inherent stochasticity of the policy in SAC, it also
    winds up benefiting from something like target policy smoothing.
    A central feature of SAC is entropy regularization. The policy is trained to maximize a trade-off
    between expected return and entropy, a measure of randomness in the policy. This has a close
    connection to the exploration-exploitation trade-off: increasing entropy results in more exploration,
    which can accelerate learning later on. It can also prevent the policy from prematurely converging
    to a bad local optimum.
    """
    def __init__(self,obs_dim,act_dim,hidden_sizes,act_limit,gamma,polyak,pi_lr,q_lr,alpha_lr,alpha):
        self.pi = gaussian_actor_model(obs_dim,act_dim,hidden_sizes,'relu')
        self.q = twin_critic_model(obs_dim,act_dim,hidden_sizes,'relu')
        self.q_target = deepcopy(self.q)
        self._log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp)
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.alpha_optimizer = tf.keras.optimizers.Adam(alpha_lr)
        self.gamma = gamma
        self.polyak = polyak
        self.act_limit = act_limit
        self.target_ent = -np.prod(act_dim) # heuristic
        self.alpha = alpha # fixed entropy temperature

    def policy(self, obs):
        state = tf.expand_dims(tf.convert_to_tensor(obs),0)
        mu, logstd = self.pi(state)
        act, logp = self.sample_action(mu, logstd)
        legal_act = np.clip(tf.squeeze(act).numpy(), -self.act_limit, self.act_limit)
        return legal_act

    def sample_action(self, mu, logstd):
        logstd = tf.clip_by_value(logstd,-20,2)
        std = tf.math.exp(logstd)
        dist = tfp.distributions.Normal(loc=mu,scale=std)
        action = tf.tanh(dist.sample())
        logp = tf.reduce_sum(dist.log_prob(action), axis=-1)
        logp -= tf.math.reduce_sum(2*(np.log(2)-action-tf.math.softplus(-2*action)), axis=-1)
        return action, logp

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
        update Q-function,  Like TD3, learn two Q-function and use the smaller one of two Q values
        """
        with tf.GradientTape() as tape:
            """
            Unlike TD3, use current policy to get next action
            """
            mu,logstd = self.pi(nobs, training=True)
            nact,logp = self.sample_action(mu,logstd)
            target_q1, target_q2 = self.q_target([nobs,nact], training=True)
            target_q = tf.minimum(target_q1,target_q2)-self.alpha*logp
            actual_q = rew + (1-done) * self.gamma * target_q
            pred_q1,pred_q2 = self.q([obs, act], training=True)
            q_loss = tf.keras.losses.MSE(actual_q,pred_q1) + tf.keras.losses.MSE(actual_q,pred_q2)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))
        """
        update policy
        """
        with tf.GradientTape() as tape:
            mu,logstd = self.pi(obs, training=True)
            action,logp = self.sample_action(mu,logstd)
            q1,q2 = self.q([obs, action], training=True)
            pi_loss = tf.math.reduce_mean(self.alpha*logp-tf.minimum(q1,q2))
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
        """
        update target network
        """
        copy_network_variables(self.q_target.variables, self.q.variables, self.polyak)
        """
        update alpha
        """
        with tf.GradientTape() as tape:
            mu,logstd = self.pi(obs, training=True)
            _,logp = self.sample_action(mu,logstd)
            alpha_loss = -tf.math.reduce_mean(self._alpha*logp + self.target_ent)
        alpha_grad = tape.gradient(alpha_loss, [self._log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self._log_alpha]))
        self.alpha = self._alpha.numpy()
