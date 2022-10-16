import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .core import *

def mlp_model(input_dim, output_dim, hidden_sizes, activation, output_activation):
    """
    multiple layer perception model
    """
    input = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_sizes[0], activation=activation)(input)
    for i in range(1, len(hidden_sizes)):
        x = layers.Dense(hidden_sizes[i], activation=activation)(x)
    output = layers.Dense(output_dim, activation=output_activation)(x)
    return tf.keras.Model(input, output)

class ReplayBuffer:
    """
    Replay Buffer for Policy Optimization, store experiences and calculate total rewards, advanteges
    the buffer will be used for update the policy
    """
    def __init__(self, obs_dim, act_dim, capacity, gamma=0.99, lamda=0.95):
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32) # one hot action list
        self.prob_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)

        self.gamma, self.lamda = gamma, lamda
        self.ptr, self.traj_idx = 0, 0

    def store(self, obs, act, rew, val, prob):
        self.obs_buf[self.ptr]=obs
        self.act_buf[self.ptr]=act
        self.rew_buf[self.ptr]=rew
        self.val_buf[self.ptr]=val
        self.prob_buf[self.ptr]=prob
        self.ptr += 1

    def finish_trajectory(self, last_value = 0):
        """
        For each epidode, calculating the total reward and advanteges with specific
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
            prob=self.prob_buf[s],
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
    def __init__(self,obs_dim,act_dim,hidden_sizes,clip_ratio,actor_lr,critic_lr,beta):
        self.pi = mlp_model(obs_dim,act_dim,hidden_sizes,'relu','softmax')
        self.q = mlp_model(obs_dim,1,hidden_sizes,'relu','linear')
        self.compile_models(actor_lr, critic_lr)
        self.clip_r = clip_ratio
        self.beta = beta
        self.act_dim = act_dim

    def compile_models(self, actor_lr, critic_lr):
        self.pi.compile(loss=self.actor_loss, optimizer=tf.keras.optimizers.Adam(actor_lr))
        self.q.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(critic_lr))
        print(self.pi.summary())
        print(self.q.summary())

    def policy(self, obs):
        state = tf.expand_dims(tf.convert_to_tensor(obs),0)
        pred = tf.squeeze(self.pi(state),axis=0).numpy()
        action = np.random.choice(self.act_dim, p=pred) # index of action
        value = tf.squeeze(self.q(state), axis=0).numpy()[0]
        return action, pred, value

    def value(self, obs):
        state = tf.expand_dims(tf.convert_to_tensor(obs),0)
        value = tf.squeeze(self.q(state), axis=0).numpy()[0]
        return value

    def actor_loss(self, y, y_pred):
        # y: np.hstack([advantages, probs, actions]), y_pred: predict actions
        advs, prob, acts = y[:,:1], y[:,1:1+self.act_dim],y[:,1+self.act_dim:]
        ratio = (y_pred*acts)/(prob*acts+1e-10)
        clip_adv = tf.clip_by_value(ratio, 1-self.clip_r, 1+self.clip_r)*advs
        ent = -y_pred*tf.math.log(y_pred+1e-10) #entropy loss for promote action diversity
        obj = tf.math.minimum(ratio*advs, clip_adv)+self.beta*ent
        loss = -tf.math.reduce_mean(obj)
        return loss

    def critic_loss(self, y, y_pred):
        loss = tf.keras.losses.MSE(y, y_pred)
        return loss

    def learn(self, buffer, batch_size=64, actor_iter=80, critic_iter=80):
        data = buffer.get()
        obs_buf = data['obs']
        act_buf = np.vstack(data['act'])
        ret_buf = np.vstack(data['ret'])
        adv_buf = np.vstack(data['adv'])
        prob_buf = np.vstack(data['prob'])
        self.pi.fit(
            x = obs_buf,
            y = np.hstack([adv_buf, prob_buf, act_buf]),
            batch_size = batch_size,
            epochs = actor_iter,
            shuffle = True,
            verbose = 0,
            callbacks=None
        ) # traning pi network
        self.q.fit(
            x = obs_buf,
            y = ret_buf,
            batch_size = batch_size,
            epochs = critic_iter,
            shuffle = True,
            verbose = 0,
            callbacks=None
        ) # training q network
