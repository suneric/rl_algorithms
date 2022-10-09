import numpy as np
import tensorflow as tf
from .core import *
from copy import deepcopy

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
class DQN:
    def __init__(self, obs_dim, act_dim, hidden_sizes, gamma, lr, update_stable_freq):
        self.train = mlp_model(obs_dim, act_dim, hidden_sizes,'relu','linear')
        self.stable = deepcopy(self.train)
        print(self.train.summary())
        print(self.stable.summary())
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
        self.update_policy(obs_batch, act_batch, rew_batch, nobs_batch, done_batch)

    def update_policy(self, obs, act, rew, nobs, done):
        self.learn_iter += 1
        """
        OPtimal Q-function follows Bellman Equation:
        Q*(s,a) = E [r + gamma*max(Q*(s',a'))]
        """
        with tf.GradientTape() as tape:
            # compute current Q
            val = self.train(obs) # state value
            oh_act = tf.one_hot(act, depth=self.act_dim)
            q = tf.math.reduce_sum(tf.multiply(val,oh_act), axis=-1)
            # compute target Q
            nval, nact = self.stable(nobs), tf.math.argmax(self.train(nobs),axis=-1)
            oh_nact = tf.one_hot(nact, depth=self.act_dim)
            next_q = tf.math.reduce_sum(tf.math.multiply(nval,oh_nact), axis=-1)
            target_q = rew+self.gamma*(1-done)*next_q
            loss = tf.keras.losses.MSE(target_q, q)
        grad = tape.gradient(loss, self.train.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.train.trainable_variables))

        """
        copy train network weights to stable network
        """
        if self.learn_iter % self.update_stable_freq == 0:
            copy_network_variables(self.stable.trainable_variables, self.train.trainable_variables)
