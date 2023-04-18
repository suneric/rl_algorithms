import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import deque
import tensorflow_probability as tfp
import scipy.signal

np.random.seed(123)
tf.random.set_seed(123)

def smoothExponential(data, weight):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def copy_network_variables(target_weights, from_weights, polyak = 0.0):
    for (a,b) in zip(target_weights, from_weights):
        a.assign(a*polyak + b*(1-polyak))

def build_mlp_model(n_hidden,n_input,n_output):
    input = layers.Input(shape=(n_input,))
    x = layers.Dense(n_hidden, activation='relu')(input)
    x = layers.Dense(n_hidden, activation='relu')(x)
    output = layers.Dense(n_output,activation='linear')(x)
    model = keras.Model(input,output,name='mlp')
    print(model.summary())
    return model

def build_rnn_model(n_hidden,n_input,n_output,seq_len):
    input = layers.Input(shape=(seq_len,n_input)) # given a fixed seqence length
    x = layers.LSTM(n_hidden,activation='relu',return_sequences=True)(input)
    x = layers.LSTM(n_hidden,activation='relu',return_sequences=False)(x)
    output = layers.Dense(n_output,activation='linear')(x)
    model = keras.Model(input,output,name='rnn')
    print(model.summary())
    return model

class RolloutBuffer:
    def __init__(self,obs_dim,capacity,recurrent=False):
        self.s_buf = np.zeros((capacity,obs_dim), dtype=np.float32)
        self.s1_buf = np.zeros((capacity,obs_dim),dtype=np.float32)
        self.a_buf = np.zeros(capacity,dtype=np.int32)
        self.r_buf = np.zeros(capacity,dtype=np.float32)
        self.d_buf = np.zeros(capacity,dtype=np.float32)
        self.ptr, self.size, self.capacity = 0, 0, capacity
        self.obs_dim = obs_dim
        self.recurrent = recurrent

    def add_sample(self,s,a,r,s1,d):
        self.s_buf[self.ptr] = s
        self.s1_buf[self.ptr] = s1
        self.a_buf[self.ptr] = a
        self.r_buf[self.ptr] = r
        self.d_buf[self.ptr] = d
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self,batch_size=32,seq_len=1):
        idxs = np.random.choice(self.size, batch_size)
        s_batch = np.zeros((batch_size,seq_len,self.obs_dim),dtype=np.float32)
        s1_batch = np.zeros((batch_size,seq_len,self.obs_dim),dtype=np.float32)
        if not self.recurrent:
            s_batch = self.s_buf[idxs]
            s1_batch = self.s1_buf[idxs]
        else: # batch sequence sample
            for i in range(batch_size):
                s_batch[i] = self.s_buf[np.r_[idxs[i]-seq_len:idxs[i]]]
                s1_batch[i] = self.s1_buf[np.r_[idxs[i]-seq_len:idxs[i]]]
        s_batch = tf.convert_to_tensor(s_batch)
        s1_batch = tf.convert_to_tensor(s1_batch)
        a_batch = tf.convert_to_tensor(self.a_buf[idxs])
        r_batch = tf.convert_to_tensor(self.r_buf[idxs])
        d_batch = tf.convert_to_tensor(self.d_buf[idxs])
        return (s_batch,s1_batch,a_batch,r_batch,d_batch)

class DQNAgent:
    def __init__(self,model,gamma,lr,update_freq,act_dim):
        self.q = model
        self.q_stable = deepcopy(self.q)
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.update_freq = update_freq
        self.learn_iter = 0
        self.act_dim = act_dim

    def policy(self, obs, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.act_dim)
        else:
            return np.argmax(self.q(np.expand_dims(obs,axis=0)))

    def learn(self, buffer, batch_size=32, epoch=1):
        """
        Q*(s,a) = E [r + gamma*max(Q*(s',a'))]
        """
        self.learn_iter += 1
        (o,o1,a,r,d) = buffer.sample(batch_size)
        epoch_loss = np.zeros(epoch)
        for i in range(epoch):
            with tf.GradientTape() as tape:
                tape.watch(self.q.trainable_variables)
                # compute current Q
                oh_a = tf.one_hot(a,depth=self.act_dim)
                pred_q = tf.math.reduce_sum(self.q(o)*oh_a,axis=-1)
                # compute target Q
                oh_a1 = tf.one_hot(tf.math.argmax(self.q(o1),axis=-1),depth=self.act_dim)
                next_q = tf.math.reduce_sum(self.q_stable(o1)*oh_a1,axis=-1)
                true_q = r + (1-d) * self.gamma * next_q
                loss = tf.keras.losses.MSE(true_q, pred_q)
            grad = tape.gradient(loss, self.q.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.q.trainable_variables))
            epoch_loss[i] = loss
        #print("learning mean loss: {:.4f}".format(np.mean(epoch_loss)))
        if self.learn_iter % self.update_freq == 0:
            copy_network_variables(self.q_stable.trainable_variables, self.q.trainable_variables)

def dqn_train(env, max_eps=100):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("state {}, action {}".format(obs_dim,act_dim))
    m = build_mlp_model(n_hidden=64,n_input=obs_dim,n_output=act_dim)
    buffer = RolloutBuffer(obs_dim,50000,recurrent=False)
    agent = DQNAgent(model=m,gamma=0.99,lr=3e-4,update_freq=100,act_dim=act_dim)
    epsilon, epsilon_stop, decay = 1.0, 0.1, 0.99
    t, update_after, max_steps = 0, 1e3, 500
    ep_returns, avg_returns = [], []
    for ep in range(max_eps):
        epsilon = max(epsilon_stop, epsilon*decay)
        state = env.reset()
        step, done, ep_ret = 1, False, 0
        while not done and step <= max_steps:
            o = state[0]
            if t > update_after:
                a = agent.policy(o, epsilon)
            else:
                a = np.random.randint(act_dim)
            next_state = env.step(a)
            o1, r, d = next_state[0], next_state[1], next_state[2]
            state = next_state
            ep_ret += r
            step += 1
            t += 1
            buffer.add_sample(o,a,r,o1,d)
            if t > update_after:
                agent.learn(buffer,batch_size=16,epoch=1)
        # update
        ep_returns.append(ep_ret)
        avg_ret = np.mean(ep_returns[-20:])
        avg_returns.append(avg_ret)
        print("{}th episodic total reward: {:.4f}, total steps {}".format(ep+1, avg_ret, t))
    return avg_returns

class RDQNAgent:
    def __init__(self, model, gamma, lr, update_freq, act_dim, seq_len):
        self.q = model
        self.q_stable = deepcopy(self.q)
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.update_freq = update_freq
        self.learn_iter = 0
        self.act_dim = act_dim
        self.seq_len = seq_len

    def policy(self, obs_seq, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.act_dim)
        else:
            dim = np.array(obs_seq).shape
            digit = self.q(np.reshape(np.array(obs_seq),[1,dim[0],dim[1]]))
            return np.argmax(digit)

    def learn(self,buffer,batch_size=32, epoch=1):
        """
        Q*(s,a) = E [r + gamma*max(Q*(s',a'))]
        """
        self.learn_iter += 1
        (o,o1,a,r,d) = buffer.sample(batch_size=batch_size,seq_len=self.seq_len)
        for _ in range(epoch):
            with tf.GradientTape() as tape:
                tape.watch(self.q.trainable_variables)
                # compute current Q
                oh_a = tf.one_hot(a,depth=self.act_dim)
                pred_q = tf.math.reduce_sum(self.q(o)*oh_a,axis=-1)
                # compute target Q
                oh_a1 = tf.one_hot(tf.math.argmax(self.q(o1),axis=-1),depth=self.act_dim)
                next_q = tf.math.reduce_sum(self.q_stable(o1)*oh_a1,axis=-1)
                true_q = r + (1-d) * self.gamma * next_q
                loss = tf.keras.losses.MSE(true_q, pred_q)
            grad = tape.gradient(loss, self.q.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.q.trainable_variables))
            print("learning epoch loss: {:.4f}".format(loss))
            """
            copy train network weights to stable network
            """
        if self.learn_iter % self.update_freq == 0:
            copy_network_variables(self.q_stable.trainable_variables, self.q.trainable_variables)

def rdqn_train(env,max_eps=100):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("state {}, action {}".format(obs_dim,act_dim))
    seq_len = 16
    m = build_rnn_model(n_hidden=seq_len,n_input=obs_dim,n_output=act_dim,seq_len=seq_len)
    buffer = RolloutBuffer(obs_dim,50000,recurrent=True)
    agent = RDQNAgent(model=m,gamma=0.99,lr=3e-4,update_freq=100,act_dim=act_dim,seq_len=seq_len)
    epsilon, epsilon_stop, decay = 1.0, 0.1, 0.99
    t, update_after, max_steps = 0, 1e4, 500
    ep_returns, avg_returns = [], []
    for ep in range(max_eps):
        epsilon = max(epsilon_stop, epsilon*decay)
        state = env.reset()
        step, done, ep_ret = 1, False, 0
        o_seq = deque(maxlen=seq_len)
        while not done and step <= max_steps:
            o = state[0]
            o_seq.append(o)
            if t > update_after:
                a = agent.policy(o_seq, epsilon)
            else:
                a = np.random.randint(act_dim)
            next_state = env.step(a)
            o1, r, d = next_state[0], next_state[1], next_state[2]
            o1_seq = o_seq.copy()
            o1_seq.append(o1)
            state = next_state
            o_seq = o1_seq
            ep_ret += r
            step += 1
            t += 1
            buffer.add_sample(o,a,r,o1,d)
        # update
        if t > update_after:
            for _ in range(20):
                agent.learn(buffer,batch_size=32,epoch=10)
        ep_returns.append(ep_ret)
        avg_ret = np.mean(ep_returns[-20:])
        avg_returns.append(avg_ret)
        print("{}th episodic total reward: {:.4f}, total steps {}".format(ep+1, avg_ret, t))
    return avg_returns

if __name__ == '__main__':
    env = gym.make("LunarLander-v2", continuous=False, render_mode='human')
    ep_returns = rdqn_train(env,500)
    plt.plot(ep_returns)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    env.close()
