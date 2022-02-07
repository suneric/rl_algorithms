"""
DQN of tensorflow implementation

Reinforcement learning (RL) is an area of machine learning that is focused on
training agents to take certain actions at certain states from within an
environment to maximize total rewards. DQN is a RL algorithm where a deep
learning model is built to find the actions an agent can take at each state.

Techniqucal Defintions
- current state: s
- next state: s'
- action: a
- policy: p
- reward: r
- state-action value function: Q(s,a) is the expected total reward from an agent
starting from the current state and the output of it is known as the Q value.

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
import gym
import os
import datetime
from statistics import mean
from gym import wrappers
import argparse
"""
An off-policy replay memory for DQN agent
"""
class ReplayMemory:
    def __init__(self, memorySize, minibatchSize):
        self.memorySize = memorySize
        self.minibatchSize = minibatchSize
        self.experience = [None]*self.memorySize
        self.index = 0
        self.size = 0

    """
    store experience as a tuple (s, a, r, s', done)
    """
    def store(self, obs, act, rew, newobs, done):
        self.experience[self.index] = (obs, act, rew, newobs, done)
        # if the index is greater than memory size, flush the index
        self.index = (self.index + 1) % self.memorySize
        self.size = min(self.size+1, self.memorySize)

    """
    sampling the minibatch of experience
    return is a dict of (s,a,r,ns,done)
    """
    def sample(self):
        ids = np.random.randint(0, self.size, size=self.minibatchSize)
        batch = dict(
            obs = np.asarray([self.experience[int(id)][0] for id in ids]),
            act = np.asarray([self.experience[int(id)][1] for id in ids]),
            rew = np.asarray([self.experience[int(id)][2] for id in ids]),
            nobs = np.asarray([self.experience[int(id)][3] for id in ids]),
            dones = np.asarray([self.experience[int(id)][4] for id in ids]),
        )
        return batch

"""
NN model for policy
take observation state as input and output actions
using TF2 autograph in tf.function(), create a model calss by subclassing Keras
"""
class MLPModel(tf.keras.Model):
    def __init__(self, numStates, hiddenUnits, numActions):
        super(MLPModel, self).__init__()
        self.inputLayer = tf.keras.layers.InputLayer(input_shape=(numStates,))
        self.hiddenLayers = []
        for i in hiddenUnits:
            self.hiddenLayers.append(tf.keras.layers.Dense(i, activation='relu',kernel_initializer='RandomNormal'))
        self.outputLayer = tf.keras.layers.Dense(numActions, activation='linear',kernel_initializer='RandomNormal')

    """
    model forward pass
    input shape is [batch size, state dimension]
    output shape is [batch size, action dimension]
    The @tf.function annotation of call() enables autograph and automatic control
    dependencies
    """
    @tf.function
    def call(self, inputs):
        z = self.inputLayer(inputs)
        for layer in self.hiddenLayers:
            z = layer(z)
        output = self.outputLayer(z)
        return output

"""
Deep Q-learning Network Agent
"""
class DQNAgent:
    def __init__(self, numStates, numActions, hiddenUnits, gamma, memorySize, batchSize, lr):
        self.numActions = numActions
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.trainNN = MLPModel(numStates, hiddenUnits, numActions)
        self.stableNN = MLPModel(numStates, hiddenUnits, numActions)
        self.replayBuffer = ReplayMemory(memorySize, batchSize)
        self.lossFn = tf.keras.losses.MeanSquaredError()
        # self.trainNN.summary()
        # self.stableNN.summary()

    def train(self):
        minibatch = self.replayBuffer.sample()
        with tf.GradientTape() as tape:
            # current Q
            vals = self.trainNN(minibatch['obs'])
            acts = minibatch['act']
            ohActs = tf.one_hot(acts,depth=self.numActions)
            predQ = tf.math.reduce_sum(tf.math.multiply(vals,ohActs), axis=-1) # -1:last axis
            # next Q
            nVals = self.stableNN(minibatch['nobs'])
            nActs = tf.math.argmax(self.trainNN(minibatch['nobs']),axis=-1)
            ohnActs = tf.one_hot(nActs, depth=self.numActions)
            nextQ = tf.math.reduce_sum(tf.math.multiply(nVals,ohnActs), axis=-1)
            # target Q using Bellman Equation: Q(s,a) = max(r+Q(s',a))
            rewards = minibatch['rew']
            dones = minibatch['dones']
            actualQ = np.where(dones, rewards, rewards+self.gamma*nextQ)
            # compute loss
            lossQ = self.lossFn(y_true=actualQ, y_pred=predQ)
        # gradient decent
        gradients = tape.gradient(lossQ, self.trainNN.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainNN.trainable_weights))
        return lossQ

    """
    get action based on epsilon greedy
    """
    def action_eg(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.numActions)
        else:
            vals = self.trainNN(np.atleast_2d(states))[0]
            return np.argmax(vals)

    """
    copy trainable weights from train network to stable network
    """
    def copy_weights(self):
        self.stableNN.set_weights(self.trainNN.get_weights())

########
np.random.seed(123)

def play_game(env, agent, epsilon, syncStep):
    rewards = 0
    iter = 0
    done = False
    obs = env.reset()
    losses = []
    while not done: # an episode
        act = agent.action_eg(obs,epsilon)
        prevObs = obs
        obs, rew, done, _ = env.step(act)
        rewards += rew
        if done:
            env.reset()
        # store transitions
        agent.replayBuffer.store(prevObs, act, rew, obs, done)
        loss = agent.train()
        losses.append(loss.numpy())
        iter += 1
        # sync two networks
        if iter % syncStep == 0:
            agent.copy_weights()

    return rewards, mean(losses)

def make_video(env, agent):
    env = wrappers.Monitor(env,os.path.join(os.getcwd(),"videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    obs = env.reset()
    while not done:
        env.render()
        act = agent.action_eg(obs,0)
        obs, rew, done, _ = env.step(act)
        steps += 1
        rewards += rew
    print("Test Step {} Rewards {}".format(steps, rewards))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ep', type=int, default=100)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    maxEpisode = args.max_ep
    syncStep = 50
    epsilon = 0.99
    decay = 0.999
    minEpsilon = 0.1

    currTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logDir = 'logs/dqn' + currTime
    summaryWriter = tf.summary.create_file_writer(logDir)

    env = gym.make('CartPole-v0')
    numStates = len(env.observation_space.sample())
    numActions = env.action_space.n
    hiddenUnits = [64,64]
    gamma = 0.99
    memorySize = 10000
    batchSize = 64
    lr = 1e-3
    agent = DQNAgent(numStates,numActions,hiddenUnits,gamma,memorySize,batchSize,lr)

    totalRewards = np.empty(maxEpisode)
    for i in range(maxEpisode):
        epsilon = max(minEpsilon, epsilon*decay)
        totalR, losses = play_game(env, agent, epsilon, syncStep)
        totalRewards[i] = totalR
        averageR = totalRewards[max(0,i-100):(i+1)].mean()
        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', totalR, step=i)
            tf.summary.scalar('average loss', losses, step=i)
        if i%100 == 0:
            print("Episode: {}  Average Rewards: {:.4f}  Epsilon: {:.4f}  Losses: {:.4f}".format(i,averageR,epsilon,losses))

    make_video(env, agent)

    env.close()
