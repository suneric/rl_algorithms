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

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from IPython import display
from time import sleep

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
Limiting GPU memory growth
"""
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

"""
Memory Buffer for storing experiences
All standard algorithm for training a DNN to approximator Q*(s,a) make use of an experience replay buffer.
This is the set D of previous experiences. In order for the algorithm to have stable behavior, the replay
buffer should be large enough to contain a wide range of experiences, but it may not always be good to keep
everything. If you only use the very-most recent data, you will overfit to that and things will break; if
you use too much experience, you may slow down your learning. This may take some tuning to get right.
"""
class MemoryBuffer:
    def __init__(self,capacity,batchSize,numStates,numActions):
        self.capacity = capacity
        self.batchSize = batchSize
        self.stateBuffer = np.zeros((self.capacity,numStates),dtype=np.float32)
        self.actionBuffer = np.zeros((self.capacity,numActions),dtype=np.float32)
        self.rewardBuffer = np.zeros((self.capacity,1),dtype=np.float32)
        self.nextStateBuffer = np.zeros((self.capacity,numStates),dtype=np.float32)
        self.counter = 0

    """
    Takes (s,a,r,s') observation tuple as input
    """
    def record(self, obsTuple):
        index = self.counter % self.capacity
        self.stateBuffer[index] = obsTuple[0]
        self.actionBuffer[index] = obsTuple[1]
        self.rewardBuffer[index] = obsTuple[2]
        self.nextStateBuffer[index] = obsTuple[3]
        self.counter += 1

    """
    Sampling
    """
    def sample(self):
        # randomly select indices
        range = min(self.counter,self.capacity)
        batchIndices = np.random.choice(range,self.batchSize)
        # convert to tensors
        stateBatch = tf.convert_to_tensor(self.stateBuffer[batchIndices])
        actionBatch = tf.convert_to_tensor(self.actionBuffer[batchIndices])
        rewardBatch = tf.convert_to_tensor(self.rewardBuffer[batchIndices])
        nextStateBatch = tf.convert_to_tensor(self.nextStateBuffer[batchIndices])
        return dict(
            state = stateBatch,
            action = actionBatch,
            reward = rewardBatch,
            nextState = nextStateBatch
        )

"""
DDPG is an off-policy algorithm. The reason is that the Bellman equation doesn't care
which transition tuples are used, or how the actions were selected, or what happens
after a given transition, because the optimal Q-function should satisfy the Bellman
equation for all possible transition. So any transitions that we've ever experienced
are fair game when trying to fit a Q-function approximator via MSBE minimization.
"""
class DDPG:
    def __init__(self,numStates,numActions,lowerBound,upperBound,gamma,tau,lrCritic,lrActor):
        self.actionDim = numActions
        self.actor_p = self.actor_model(numStates,numActions,upperBound) # policy actor network
        self.critic_p = self.critic_model(numStates,numActions) # policy critic network
        self.actor_q = self.actor_model(numStates,numActions,upperBound) # q-function actor network: target
        self.critic_q = self.critic_model(numStates,numActions) # q-function critic network: target
        self.actor_q.set_weights(self.actor_p.get_weights())
        self.critic_q.set_weights(self.critic_p.get_weights())
        self.criticOptimizer = tf.keras.optimizers.Adam(lrCritic)
        self.actorOptimizer = tf.keras.optimizers.Adam(lrActor)
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.gamma = gamma
        self.tau = tau

    def actor_model(self, numStates, numActions, upperBound):
        lastInit = tf.random_uniform_initializer(minval=-0.003,maxval=0.003)
        inputs = layers.Input(shape=(numStates,))
        out = layers.Dense(512,activation='relu')(inputs)
        out = layers.Dense(256,activation='relu')(inputs)
        out = layers.Dense(128,activation='relu')(out)
        outputs = layers.Dense(numActions,activation='tanh',kernel_initializer=lastInit)(out)
        outputs = outputs*upperBound
        model = tf.keras.Model(inputs,outputs)
        return model

    def critic_model(self, numStates, numActions):
        stateInput = layers.Input(shape=(numStates))
        stateOut = layers.Dense(512, activation='relu')(stateInput)
        stateOut = layers.Dense(256, activation='relu')(stateOut)
        actionInput = layers.Input(shape=(numActions))
        actionOut = layers.Dense(256, activation='relu')(actionInput)
        concat = layers.Concatenate()([stateOut, actionOut])
        out = layers.Dense(256, activation='relu')(concat)
        out = layers.Dense(128, activation='relu')(out)
        outputs = layers.Dense(1)(out)
        model = tf.keras.Model([stateInput,actionInput],outputs)
        return model

    def random_policy(self):
        """
        imporve exploration at the start of training, takes actioins which are sampled from a
        uniform random distribution over valid actions
        """
        randomActions = np.random.uniform(self.lowerBound, self.upperBound, self.actionDim)
        return randomActions

    def policy(self, state, noiseObj=None):
        state = tf.expand_dims(tf.convert_to_tensor(state),0)
        sampledActions = tf.squeeze(self.actor_p(state))
        if noiseObj:
            sampledActions = sampledActions.numpy() + noiseObj()
        legalAction = np.clip(sampledActions, self.lowerBound, self.upperBound)
        return np.squeeze(legalAction)

    def learn(self, memory):
        obsBatch = memory.sample()
        stateBatch = obsBatch['state']
        actionBatch = obsBatch['action']
        rewardBatch = obsBatch['reward']
        nextStateBatch = obsBatch['nextState']
        self.update_policy(stateBatch,actionBatch,rewardBatch,nextStateBatch)
        self.update_target(self.actor_q.variables, self.actor_p.variables)
        self.update_target(self.critic_q.variables, self.critic_p.variables)

    @tf.function
    def update_policy(self, stateBatch, actionBatch, rewardBatch, nextStateBatch):
        """
        Uses off-policy data and the Bellman equation to learn the Q-function
        Q*(s,a) = E [r(s,a) + gamma x max(Q*(s',a'))]
        minimizing MSBE loss with stochastic gradient descent
        L_p = E [(Q_p(s,a) - (r + gamma x Q_q(s', u_q(s'))))^2]
        """
        with tf.GradientTape() as tape:
            targetActions = self.actor_q(nextStateBatch,training=True) # u_q(s')
            targetQ = self.critic_q([nextStateBatch, targetActions], training=True) # Q_q(s',a')
            y = rewardBatch + self.gamma * targetQ # Bellman Equation
            criticValue = self.critic_p([stateBatch,actionBatch], training=True) # Q_p(s,a)
            criticLoss = tf.math.reduce_mean(tf.math.square(y-criticValue)) # MSBE loss
        criticGrad = tape.gradient(criticLoss,self.critic_p.trainable_variables)
        self.criticOptimizer.apply_gradients(zip(criticGrad, self.critic_p.trainable_variables))

        """
        Use Q-function to learn policy
        Policy learning in DDPG is fairly simple. We want to learn a deterministic polict u(s) which
        gives the action that maximize Q(s,a). Because the action sapce is continuous, and we assume
        the Q-function is differentiable with respect to action, we can just perform gradient ascent
        to solve max(E [Q(s, u(s))])
        """
        with tf.GradientTape() as tape:
            actions = self.actor_p(stateBatch, training=True)
            criticValue = self.critic_p([stateBatch, actions], training=True)
            actorLoss = -tf.math.reduce_mean(criticValue) # '-' for gradient ascent
        actorGrad = tape.gradient(actorLoss, self.actor_p.trainable_variables)
        self.actorOptimizer.apply_gradients(zip(actorGrad, self.actor_p.trainable_variables))

    @tf.function
    def update_target(self,target_weights, weights):
        """
        In DQN-based algorithms, the target network is just copied over from the main network
        every some-fixed-number of steps. In DDPG-style algorithm, the target network is updated
        once per main network update by polyak averaging, where polyak(tau) usually close to 1.
        """
        for (a,b) in zip(target_weights, weights):
            a.assign(a*self.tau + b*(1-self.tau))


"""
Gaussian Noise added to Action for better exploration
DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the
agent were to explore on-policy, int the beginning it would probably not try a wide ennough varienty
of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their
actions at traiing time. Uncorreletaed, mean-zero Gaussian noise work perfectly well, and it is suggested
as it is simpler. At test time, to see how well the policy exploits what it has learned, we don not add
noise to the actions.
"""
class ActionNoise:
    def __init__(self, mean, std_dev, dim):
        self.mean = mean
        self.std_dev = std_dev
        self.size = dim

    def __call__(self):
        return np.random.normal(self.mean,self.std_dev,self.size)

class OUActionNoise:
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

np.random.seed(123)

if __name__ == '__main__':
    env = gym.make(
        "LunarLander-v2",
        continuous = True,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5,
        render_mode = 'human'
    )
    numStates = env.observation_space.shape[0]
    numActions = env.action_space.shape[0]
    upperBound = env.action_space.high[0]
    lowerBound = env.action_space.low[0]
    print("state dimension {}, action dimension {}, bound [{},{}]".format(
        numStates,
        numActions,
        lowerBound,
        upperBound)
        )

    memory = MemoryBuffer(50000,64,numStates,numActions)
    noise = OUActionNoise(mean=np.zeros(numActions),std_dev=0.1*upperBound)
    agent = DDPG(
        numStates = numStates,
        numActions = numActions,
        lowerBound = lowerBound,
        upperBound = upperBound,
        gamma = 0.99,
        tau = 0.995,
        lrCritic = 2e-3,
        lrActor = 1e-3
        )

    logDir = 'logs/ddpg' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaryWriter = tf.summary.create_file_writer(logDir)

    totalEpisodes = 10000
    epReturnList, avgReturnList = [], []
    for ep in range(totalEpisodes):
        obs = env.reset()
        state = obs[0]
        epReturn = 0
        while True:
            action = agent.policy(state,noise)
            obs = env.step(action)
            nextState = obs[0]
            reward = obs[1]
            done = obs[2]
            memory.record((state,action,reward,nextState))
            agent.learn(memory)
            epReturn += reward
            if done:
                break
            state = nextState

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
