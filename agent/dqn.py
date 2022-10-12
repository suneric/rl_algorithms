import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .core import *
from copy import deepcopy

def mlp_model(obs_dim, act_dim, hidden_sizes, activation):
    last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    input = layers.Input(shape=(obs_dim,))
    x = layers.Dense(hidden_sizes[0], activation=activation)(input)
    for i in range(1, len(hidden_sizes)):
        x = layers.Dense(hidden_sizes[i], activation=activation)(x)
    output = layers.Dense(act_dim, activation="linear", kernel_initializer=last_init)(x)
    model = tf.keras.Model(input, output)
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
        self.act_buf = np.zeros(capacity, dtype=np.int32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
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

class DQN:
    """
    DQN target is to approximate Q(s,a) which is updated through back propagation.
    prediction: y' = f(s,theta)
    loss: L(y,y')= L(Q(s,a), f(s,theta))
    in the back propagation process, we take the partial derivative of the loss
    function to theta to find a value of theta that minimizes the loss. The ground-
    truth Q(s,a) can be found with the Bellman Equation: Q(s,a) = max(r+Q(s',a))
    where Q(s',a) = f(s',theta), if the s' is not the terminal state, otherwise
    Q(s',a) = 0, so for the terminal state, Q(s,a) = r.

    Problems:
    Because we are using the model prediction f(s' theta) to approximate the real
    value of Q(s', a), this is called semi-gradient, which could be very unstable
    since the real target will change each time the model updates itself. The
    solution is to create target network that is essentially a copy of the traning
    model at certain steps so the target model updates less frequently.

    Another issue with the model is overfitting. When update the mode after the end
    of each game, we have already potentially played hundreds of steps, so we are
    essentially doing batch gradient descent. Because each batch always contains
    steps from one full game, the model might not learn well from it. To solve this,
    we create an experience reply buffer that stores the (s,s',a,r) values of several
    hundreds of games and randomly select a batch from it each time to update the
    model.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, gamma, lr, update_stable_freq):
        self.train = mlp_model(obs_dim, act_dim, hidden_sizes,'relu')
        self.stable = deepcopy(self.train)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.act_dim = act_dim
        self.learn_iter = 0
        self.update_stable_freq = update_stable_freq

    def policy(self, obs, epsilon):
        """
        get action based on epsilon greedy
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.act_dim)
        else:
            return np.argmax(self.train(np.expand_dims(obs, axis=0)))

    def learn(self, buffer):
        sampled_batch = buffer.sample()
        obs_batch = sampled_batch['obs']
        nobs_batch = sampled_batch['nobs']
        act_batch = sampled_batch['act']
        rew_batch = sampled_batch['rew']
        done_batch = sampled_batch['done']
        self.update(obs_batch, act_batch, rew_batch, nobs_batch, done_batch)

    @tf.function
    def update(self, obs, act, rew, nobs, done):
        self.learn_iter += 1
        """
        OPtimal Q-function follows Bellman Equation:
        Q*(s,a) = E [r + gamma*max(Q*(s',a'))]
        """
        with tf.GradientTape() as tape:
            # compute current Q
            val = self.train(obs,training=True) # state value
            oh_act = tf.one_hot(act, depth=self.act_dim)
            pred_q = tf.math.reduce_sum(tf.multiply(val,oh_act), axis=-1)
            # compute target Q
            nval, nact = self.stable(nobs), tf.math.argmax(self.train(nobs, training=True), axis=-1)
            oh_nact = tf.one_hot(nact, depth=self.act_dim)
            next_q = tf.math.reduce_sum(tf.math.multiply(nval,oh_nact), axis=-1)
            actual_q = rew + (1-done) * self.gamma * next_q
            loss = tf.keras.losses.MSE(actual_q, pred_q)
        grad = tape.gradient(loss, self.train.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.train.trainable_variables))

        """
        copy train network weights to stable network
        """
        if self.learn_iter % self.update_stable_freq == 0:
            copy_network_variables(self.stable.trainable_variables, self.train.trainable_variables)
