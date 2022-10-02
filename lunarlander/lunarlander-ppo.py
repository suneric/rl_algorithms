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
[3] https://keras.io/examples/rl/ppo_cartpole/
"""

import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import datetime
import argparse
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from gym import wrappers
import os

"""
Replay Buffer, store experiences and calculate total rewards, advanteges
the buffer will be used for update the policy
"""
class MemoryBuffer:
    def __init__(self, capacity, numStates, gamma=0.99, lamda=0.95):
        self.capacity = capacity
        self.obs_buf = np.zeros((self.capacity, numStates), dtype=np.float32) # states
        self.act_buf = np.zeros(self.capacity, dtype=np.int32) # action, based on stochasitc policy with the probability
        self.rew_buf = np.zeros(self.capacity, dtype=np.float32) # step reward
        self.ret_buf = np.zeros(self.capacity, dtype=np.float32) # ep_return, total reward of episode
        self.val_buf = np.zeros(self.capacity, dtype=np.float32) # value of (s,a), output of critic net
        self.adv_buf = np.zeros(self.capacity, dtype=np.float32) # advantege Q(s,a)-V(s)
        self.logprob_buf = np.zeros(self.capacity, dtype=np.float32) # prediction: action probability, output of actor net
        self.gamma = gamma
        self.lamda = lamda
        self.counter = 0
        self.idx = 0

    def record(self, observation, action, reward, value, logprob):
        index = self.counter % self.capacity
        self.obs_buf[index]=observation
        self.act_buf[index]=action
        self.rew_buf[index]=reward
        self.val_buf[index]=value
        self.logprob_buf[index]=logprob
        self.counter += 1

    def get(self):
        """
        get all data of the buffer and normalize the advantages
        """
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf-adv_mean)/adv_std
        return dict(
            states=self.obs_buf,
            actions=self.act_buf,
            advantages=self.adv_buf,
            returns=self.ret_buf,
            logprobs=self.logprob_buf,
        )
        self.counter, self.idx = 0, 0

    def update(self, lastValue = 0):
        """
        For each epidode, calculating the total reward and advanteges with specific
        magic from rllab for computing discounted cumulative sums of vectors
        input: vector x: [x0, x1, x2]
        output: [x0+discount*x1+discount^2*x2, x1+discount*x2, x2]
        """
        def discount_cumsum(x,discount):
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
        ep_slice = slice(self.idx, self.counter)
        rews = np.append(self.rew_buf[ep_slice], lastValue)
        vals = np.append(self.val_buf[ep_slice], lastValue)
        deltas = rews[:-1]+self.gamma*vals[1:]-vals[:-1]
        self.adv_buf[ep_slice] = discount_cumsum(deltas, self.gamma*self.lamda) # General Advantege Estimation
        self.ret_buf[ep_slice] = discount_cumsum(rews, self.gamma)[:-1] # rewards-to-go, which is targets for the value function
        self.idx = self.counter

"""
loss print call back
"""
class PrintLoss(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        print("epoch index", epoch+1, "loss", logs.get('loss'))

def logprobabilities(logits, action, numActions):
    logprob_all = tf.nn.log_softmax(logits)
    logprob = tf.reduce_sum(tf.one_hot(action, numActions)*logprob_all, axis=1)
    return logprob

"""
Actor net
"""
class ActorModel:
    def __init__(self, numStates, numActions, clipRatio, lr):
        self.numStates = numStates
        self.numActions = numActions
        self.clipRatio = clipRatio
        self.policyNN = self.build_model()
        self.lossPrinter = PrintLoss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def build_model(self):
        inputs = layers.Input(shape=(self.numStates,))
        out = layers.Dense(512,activation='relu')(inputs)
        out = layers.Dense(256,activation='relu')(inputs)
        out = layers.Dense(128,activation='relu')(out)
        outputs = layers.Dense(self.numActions,activation=None)(out)
        model = tf.keras.Model(inputs,outputs)
        return model

    def predict(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state),0)
        logits = self.policyNN(state)
        action = tf.squeeze(tf.random.categorical(logits, 1),axis=1)
        return logits, action

    @tf.function
    def train_policy(self, obs_buf, act_buf, logprob_buf, adv_buf):
        with tf.GradientTape() as tape:
            logits = self.policyNN(obs_buf)
            ratio = tf.exp(logprobabilities(logits, act_buf, self.numActions)-logprob_buf)
            minAdv = tf.where(adv_buf > 0, (1+self.clipRatio)*adv_buf, (1-self.clipRatio)*adv_buf)
            policyLoss = -tf.reduce_mean(tf.minimum(ratio*adv_buf, minAdv))
        policyGrads = tape.gradient(policyLoss, self.policyNN.trainable_variables)
        self.optimizer.apply_gradients(zip(policyGrads, self.policyNN.trainable_variables))
        k1 = tf.reduce_mean(logprob_buf - logprobabilities(self.policyNN(obs_buf), act_buf, self.numActions))
        k1 = tf.reduce_sum(k1)
        return k1

"""
Critic net
"""
class CriticModel:
    def __init__(self, numStates, numActions, lr):
        self.numStates = numStates
        self.numActions = numActions
        self.valueNN = self.build_model()
        self.lossPrinter = PrintLoss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def build_model(self):
        inputs = layers.Input(shape=(self.numStates,))
        out = layers.Dense(512,activation='relu')(inputs)
        out = layers.Dense(256,activation='relu')(inputs)
        out = layers.Dense(128,activation='relu')(out)
        outputs = layers.Dense(1,activation=None)(out)
        model = tf.keras.Model(inputs,outputs)
        return model

    def predict(self,state):
        state = tf.expand_dims(tf.convert_to_tensor(state),0)
        digits = self.valueNN(state)
        value = tf.squeeze(digits, axis=1)
        return value

    @tf.function
    def train_value(self, obs_buf, ret_buf):
        with tf.GradientTape() as tape:
            valueLoss = tf.reduce_mean((ret_buf - self.valueNN(obs_buf)) ** 2)
        valueGrads = tape.gradient(valueLoss, self.valueNN.trainable_variables)
        self.optimizer.apply_gradients(zip(valueGrads, self.valueNN.trainable_variables))

"""
PPO Agent
"""
class PPOAgent:
    def __init__(self, numStates, numActions, clipRatio, policyLR, valueLR, targetK1):
        self.actor = ActorModel(numStates, numActions, clipRatio, policyLR)
        self.critic = CriticModel(numStates, numActions, valueLR)
        self.actDim = numActions
        self.targetK1 = targetK1

    def action(self, state):
        logits, action = self.actor.predict(state) # sample action from actor
        logprob = logprobabilities(logits, action, self.actDim) # get log-probability
        value = self.critic.predict(state) # get value
        return logprob, action, value

    def train(self, replyBuffer, itActor=80, itCritic=80):
        data = replyBuffer.get()
        obs_buf = data['states']
        act_buf = data['actions']
        adv_buf = data['advantages']
        ret_buf = data['returns']
        logprob_buf = data['logprobs']
        for _ in range(itActor):
            k1 = self.actor.train_policy(obs_buf, act_buf, logprob_buf, adv_buf)
            if k1 > 1.5 * self.targetK1:
                break # Early Stopping
        # train value network
        for _ in range(itCritic):
            self.critic.train_value(obs_buf, ret_buf)

np.random.seed(123)

if __name__ == '__main__':
    env = gym.make(
        "LunarLander-v2",
        continuous = False,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5,
        render_mode = 'human'
    )
    numStates = env.observation_space.shape[0]
    numActions = env.action_space.n
    print("state dimension {}, action dimension {}".format(
        numStates,
        numActions)
        )

    logDir = 'logs/ppo' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    buffer = MemoryBuffer(capacity=5000, numStates = numStates, gamma=0.99, lamda=0.97)
    agent = PPOAgent(numStates = numStates, numActions = numActions, clipRatio = 0.2, policyLR = 3e-4, valueLR = 1e-3, targetK1 = 0.01)

    totalEpisodes = 1000
    epReturnList, avgReturnList = [], []
    for ep in range(totalEpisodes):
        obs = env.reset()
        state = obs[0]
        epReturn = 0
        while True:
            logprob, action, value = agent.action(state)
            obs = env.step(action[0].numpy())
            newState = obs[0]
            reward = obs[1]
            done = obs[2]
            buffer.record(state, action, reward, value, logprob)
            epReturn += reward
            if done:
                break
            state = newState

        buffer.update()
        agent.train(replyBuffer=buffer, itActor= 100, itCritic= 100)
        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', epReturn, step=ep)

        epReturnList.append(epReturn)
        avgReward = np.mean(epReturnList[-40:])
        print("Episode * {} * average reward is ===> {}".format(ep, avgReward))
        avgReturnList.append(avgReward)

    env.close()

    plt.plot(avgReturnList)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Episodic Reward')
    plt.show()
