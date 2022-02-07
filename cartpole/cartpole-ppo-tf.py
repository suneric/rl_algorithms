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
from gym import wrappers
import os

"""
Replay Buffer, store experiences and calculate total rewards, advanteges
the buffer will be used for update the policy
"""
class ReplayBuffer:
    def __init__(self, obs_dim, size, gamma=0.99, lamda=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32) # states
        self.act_buf = np.zeros(size, dtype=np.int32) # action, based on stochasitc policy with teh probability
        self.rew_buf = np.zeros(size, dtype=np.float32) # step reward
        self.ret_buf = np.zeros(size, dtype=np.float32) # ep_return, total reward of episode
        self.val_buf = np.zeros(size, dtype=np.float32) # value of (s,a), output of critic net
        self.adv_buf = np.zeros(size, dtype=np.float32) # advantege Q(s,a)-V(s)
        self.logprob_buf = np.zeros(size, dtype=np.float32) # prediction: action probability, output of actor net
        self.gamma, self.lamda = gamma, lamda
        self.ptr, self.idx = 0, 0 # buffer ptr, and current trajectory start index

    def store(self, observation, action, reward, value, logprob):
        #print("storing", state[0].shape, action.shape, reward, prediction.shape, value.shape)
        self.obs_buf[self.ptr]=observation
        self.act_buf[self.ptr]=action
        self.rew_buf[self.ptr]=reward
        self.val_buf[self.ptr]=value
        self.logprob_buf[self.ptr]=logprob
        self.ptr += 1

    """
    For each epidode, calculating the total reward and advanteges with specific
    """
    def ep_update(self, lastValue = 0):
        """
        magic from rllab for computing discounted cumulative sums of vectors
        input: vector x: [x0, x1, x2]
        output: [x0+discount*x1+discount^2*x2, x1+discount*x2, x2]
        """
        def discount_cumsum(x,discount):
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

        ep_slice = slice(self.idx, self.ptr)
        rews = np.append(self.rew_buf[ep_slice], lastValue)
        vals = np.append(self.val_buf[ep_slice], lastValue)

        deltas = rews[:-1]+self.gamma*vals[1:]-vals[:-1]
        # General Advantege Estimation
        self.adv_buf[ep_slice] = discount_cumsum(deltas, self.gamma*self.lamda)
        # rewards-to-go, which is targets for the value function
        self.ret_buf[ep_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.idx = self.ptr

    def get(self):
        # get all data of the buffer and normalize the advantages
        self.ptr, self.idx = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf-adv_mean)/adv_std
        return dict(
            states=self.obs_buf,
            actions=self.act_buf,
            advantages=self.adv_buf,
            returns=self.ret_buf,
            logprobs=self.logprob_buf,
        )

"""
loss print call back
"""
class PrintLoss(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        print("epoch index", epoch+1, "loss", logs.get('loss'))

"""
build a feedforward neural network
"""
def mlp(obsDim, hiddenSize, numActions, outputActivation=None):
    inputs = tf.keras.Input(shape=(obsDim,), dtype=tf.float32)
    x = tf.keras.layers.Dense(units=hiddenSize[0], activation='tanh')(inputs)
    for i in range(1, len(hiddenSize)):
        x = tf.keras.layers.Dense(units=hiddenSize[i], activation='tanh')(x)
    logits = tf.keras.layers.Dense(units=numActions, activation=outputActivation)(x)
    return tf.keras.Model(inputs = inputs, outputs=logits)

def logprobabilities(logits, action, numActions):
    logprob_all = tf.nn.log_softmax(logits)
    logprob = tf.reduce_sum(tf.one_hot(action, numActions)*logprob_all, axis=1)
    return logprob

"""
Actor net
"""
class ActorModel:
    def __init__(self, obsDim, hiddenSize, numActions, clipRatio, lr):
        self.policyNN = self.build_model(obsDim, hiddenSize, numActions, lr)
        self.clipRatio = clipRatio
        self.numActions = numActions
        self.lossPrinter = PrintLoss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def build_model(self, obsDim, hiddenSize, numActions, lr):
        model = mlp(obsDim, hiddenSize, numActions)
        # model.compile(loss=self.ppo_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        # print(model.summary())
        return model

    # def ppo_loss(self, y_true, y_pred):
    #     # y_true: np.hstack([advantages, predictions, actions])
    #     advs,o_pred,acts = y_true[:,:1],y_true[:,1:1+self.numActions],y_true[:,1+self.numActions:]
    #     # print(y_pred, advs, picks, acts)
    #     prob = y_pred*acts
    #     old_prob = o_pred*acts
    #     ratio = prob/(old_prob + 1e-10)
    #     p1 = ratio*advs
    #     p2 = K.clip(ratio, 1-self.clipRatio, 1+self.clipRatio)*advs
    #     # total loss = policy loss + entropy loss (entropy loss for promote action diversity)
    #     loss = -K.mean(K.minimum(p1,p2)+self.beta*(-y_pred*K.log(y_pred+1e-10)))
    #     return loss
    # def fit(self,states,y_true,epochs,batch_size):
    #     self.actor.fit(states, y_true, epochs=epochs, verbose=0, shuffle=True, batch_size=batch_size, callbacks=[self.lossPrinter])

    def predict(self, obs):
        obs = obs.reshape(1,-1)
        logits = self.policyNN(obs)
        action = tf.squeeze(tf.random.categorical(logits, 1),axis=1)
        return logits, action

    @tf.function
    def train_policy(self, obs_buf, act_buf, logprob_buf, adv_buf):
        # Record operation for automtic differentiation
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
    def __init__(self, obsDim, hiddenSize, lr):
        self.valueNN = self.build_model(obsDim, hiddenSize, lr)
        self.lossPrinter = PrintLoss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def build_model(self, obsDim, hiddenSize, lr):
        model = mlp(obsDim, hiddenSize, 1)
        # model.compile(loss="mse",optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        # print(model.summary())
        return model

    def predict(self,obs):
        obs = obs.reshape(1,-1)
        digits = self.valueNN(obs)
        value = tf.squeeze(digits, axis=1)
        return value

    # def fit(self,states,y_true,epochs,batch_size):
    #     self.critic.fit(states, y_true, epochs=epochs, verbose=0, shuffle=True, batch_size=batch_size, callbacks=[self.lossPrinter])

    @tf.function
    def train_value(self, obs_buf, ret_buf):
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            valueLoss = tf.reduce_mean((ret_buf - self.valueNN(obs_buf)) ** 2)
        valueGrads = tape.gradient(valueLoss, self.valueNN.trainable_variables)
        self.optimizer.apply_gradients(zip(valueGrads, self.valueNN.trainable_variables))

"""
PPO Agent
"""
class PPOAgent:
    def __init__(self, obsDim, hiddenSize, numActions, clipRatio, policyLR, valueLR, memorySize, gamma, lamda, targetK1):
        self.buffer = ReplayBuffer(obsDim, memorySize, gamma, lamda)
        self.Actor = ActorModel(obsDim, hiddenSize, numActions, clipRatio, policyLR)
        self.Critic = CriticModel(obsDim, hiddenSize, valueLR)
        self.actDim = numActions
        self.targetK1 = targetK1

    def action(self, obs):
        # sample action from actor
        logits, action = self.Actor.predict(obs)
        # get log-probability of taking actins by using the logits
        logprob = logprobabilities(logits, action, self.actDim)
        # get value
        value = self.Critic.predict(obs)
        return logprob, action, value

    def train(self, itActor=80, itCritic=80):
        data = self.buffer.get()
        obs_buf = data['states']
        act_buf = data['actions']
        adv_buf = data['advantages']
        ret_buf = data['returns']
        logprob_buf = data['logprobs']
        # train polict network
        for _ in range(itActor):
            k1 = self.Actor.train_policy(obs_buf, act_buf, logprob_buf, adv_buf)
            if k1 > 1.5 * self.targetK1:
                break # Early Stopping
        # train value network
        for _ in range(itCritic):
            self.Critic.train_value(obs_buf, ret_buf)

#######
np.random.seed(123)

def make_video(env, agent):
    env = wrappers.Monitor(env,os.path.join(os.getcwd(),"videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    obs = env.reset()
    while not done:
        env.render()
        logprob, action, value = agent.action(obs)
        obs, reward, done, _ = env.step(action[0].numpy())
        steps += 1
        rewards += reward
        if done:
            env.reset()
    print("Test Step {} Rewards {}".format(steps, rewards))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ep', type=int, default=10000)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    maxEpoch = args.max_ep
    epSteps = 4000
    gamma = 0.99
    lamda = 0.97
    clipRatio = 0.2
    policyLearningRate = 3e-4
    valueLearningRate = 1e-3
    policyTrainingIteration = 80
    valueTrainingIteration = 80
    targetK1 = 0.01

    currTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logDir = 'logs/ppo' + currTime
    summaryWriter = tf.summary.create_file_writer(logDir)

    env = gym.make('CartPole-v0')
    obsDim = env.observation_space.shape[0]
    numActions = env.action_space.n
    hiddenSize = [64,64]
    agent = PPOAgent(obsDim,hiddenSize,numActions,clipRatio,policyLearningRate,valueLearningRate,epSteps,gamma,lamda,targetK1)

    obs, epReturn, epLength = env.reset(), 0, 0
    # Iteration over the number of epochs
    for ep in range(maxEpoch):
        sumReturn = 0
        sumLength = 0
        numEpisodes = 0

        # Iterate over the steps of each epoch
        for t in range(epSteps):
            logprob, action, value = agent.action(obs)
            newobs, reward, done, _ = env.step(action[0].numpy())
            epReturn += reward
            epLength += 1
            agent.buffer.store(obs, action, reward, value, logprob)
            obs = newobs

            # finish trajectory if reach to a terminal state
            if done or (t == epSteps-1):
                lastValue = 0 if done else agent.Critic.predict(obs)
                agent.buffer.ep_update(lastValue)
                sumReturn += epReturn
                sumLength += epLength
                numEpisodes += 1
                with summaryWriter.as_default():
                    tf.summary.scalar('episode reward', epReturn, step=numEpisodes)
                obs, epReturn, epLength = env.reset(), 0, 0

        # update policy and value function
        agent.train(policyTrainingIteration, valueTrainingIteration)
        print("Episode: {}  Average Rewards: {:.4f}  Mean Length {:.4f} ".format(ep+1, sumReturn/numEpisodes, sumLength/numEpisodes))

    make_video(env, agent)

    env.close()
