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

Reference:
https://github.com/VXU1230/Medium-Tutorials/blob/master/dqn/cart_pole.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .core import *
from copy import deepcopy

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

class DQN:
    def __init__(self, obs_dim, act_dim, hidden_sizes, gamma, lr):
        self.train = mlp_model(obs_dim, act_dim, hidden_sizes,'relu','linear')
        self.stable = mlp_model(obs_dim, act_dim, hidden_sizes,'relu','linear')
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.act_dim = act_dim

    """
    get action based on epsilon greedy
    """
    def policy(self, obs, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.act_dim)
        else:
            values = tf.squeeze(self.train(obs))
            return np.argmax(values)

    def learn(self, buffer):
        sampled_batch = buffer.sample()
        obs_batch = sampled_batch['obs']
        nobs_batch = sampled_batch['nobs']
        act_batch = sampled_batch['act']
        rew_batch = sampled_batch['rew']
        done_batch = sampled_batch['done']
        self.update_policy(obs_batch, act_batch, rew_batch, nobs_batch, done_batch)

    def update_policy(self, obs, act, rew, nobs, done):
        with tf.GradientTape() as tape:
            print(act)
            oh_act = tf.keras.utils.to_categorical(act, self.act_dim, dtype=np.float32)
            print(oh_act)
            pred_q = tf.math.reduce_sum(tf.multiply(self.train(obs),oh_act), axis=1)
            actual_q = rew+self.gamma*(1-done)*tf.math.argmax(self.stable(nobs), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_q-pred_q))
        grad = tape.gradient(loss, self.train.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.train.trainable_weights))

    """
    copy train network weights to stable network
    """
    def update_stable(self):
        self.stable.set_weights(self.train.get_weights())
