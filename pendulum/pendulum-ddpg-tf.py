"""
Deep Deterministic Policy Gradient (DDPG)

Deep Deterministic Policy Gradient (DDPG) is a model-free off-policy algorithm
for learning continous actions. It combines ideas from DPG (Deterministic
Policy Gradient) and DQN (Deep Q-Network). It uses Experience Replay and
slow-learning target networks from DQN, and it is based on DPG, which can
operate over continuous action spaces.

Just like the Actor-Critic method, we have two networks:
    Actor - It proposes an action given a state.
    Critic - It predicts if the action is good (positive value) or bad
            (negative value) given a state and an action.
DDPG uses two more techniques not present in the original DQN:
First, it uses two Target networks.
Why? Because it add stability to training. In short, we are learning from estimated
targets and Target networks are updated slowly, hence keeping our estimated targets
stable. Conceptually, this is like saying, "I have an idea of how to play this well,
I'm going to try it out for a bit until I find something better", as opposed to
saying "I'm going to re-learn how to play this entire game after every move".
See this StackOverflow answer.

Second, it uses Experience Replay.
We store list of tuples (state, action, reward, next_state), and instead of learning
only from recent experience, we learn from sampling all of our experience accumulated
so far.

references
- https://keras.io/examples/rl/ddpg_pendulum/
"""


import gym
import tensorflow as tf
import numpy as np
import argparse
import datetime
import argparse
from gym import wrappers
import os

"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically an Ornstein-Uhlenbeck process for genetating noise, it samples noise
from a correlated normal distribution.
"""
class OUActionNoise:
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


"""
Experience buffer
"""
class ExperienceBuffer:
    def __init__(self, buffer_capacity, num_states, num_actions):
        self.buffer_capacity = buffer_capacity
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.buffer_counter = 0

    # takes (s,a,r,s') obsercation tuple as input
    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    # batch sample experiences
    def sample(self, batch_size):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, batch_size)
        # convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        return dict(
            states = state_batch,
            actions = action_batch,
            rewards = reward_batch,
            next_states = next_state_batch,
        )

class DDPGAgent:
    def __init__(self,num_states,num_actions,lower_bound,upper_bound,actor_lr,critic_lr,gamma,tau,buffer_capacity,batch_size):
        self.actor_model = self.get_actor(num_states, upper_bound)
        self.critic_model = self.get_critic(num_states, num_actions)
        self.target_actor = self.get_actor(num_states, upper_bound)
        self.target_critic = self.get_critic(num_states, num_actions)
        # making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        # experiece buffer
        self.batch_size = batch_size
        self.buffer = ExperienceBuffer(buffer_capacity,num_states,num_actions)
        self.gamma = gamma
        self.tau = tau
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # training and updating Critic network
        # y_i = r_i + gamma*Q'(s_i+1, u'(s_i+1))
        # crtic loss: L = (1/N)*sum((y_i - Q(s_i,a_i))^2)
        """
        Critic loss - Mean Squared Error of y - Q(s, a) where y is the expected
        return as seen by the Target network, and Q(s, a) is action value predicted
        by the Critic network. y is a moving target that the critic model tries to
        achieve; we make this target stable by updating the Target model slowly.
        """
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            y = reward_batch + self.gamma * self.target_critic([next_state_batch, target_actions])
            critic_value = self.critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # training and updating Actor network
        """
        Actor loss - This is computed using the mean of the value given by the
        Critic network for the actions taken by the Actor network. We seek to
        maximize this quantity.
        """
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch)
            critic_value = self.critic_model([state_batch, actions])
            # use "-" as we want to maximize the value given by the ctic for our action
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    def learn(self):
        experiences = self.buffer.sample(self.batch_size)
        state_batch = experiences['states']
        action_batch = experiences['actions']
        reward_batch = experiences['rewards']
        next_state_batch = experiences['next_states']
        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    @tf.function
    # Based on rate 'tau', which is much less than one, this update target parameters slowly
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b*self.tau + a*(1-self.tau))

    def get_actor(self, num_states, upper_bound):
        last_init = tf.random_uniform_initializer(minval=-0.003,maxval=0.003)
        inputs = tf.keras.layers.Input(shape=(num_states,))
        x = tf.keras.layers.Dense(256, activation="relu")(inputs)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=last_init)(x)
        outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self, num_states, num_actions):
        # state input
        state_input = tf.keras.layers.Input(shape=(num_states))
        state_out = tf.keras.layers.Dense(16, activation="relu")(state_input)
        state_out = tf.keras.layers.Dense(32, activation="relu")(state_out)
        # action input
        action_input = tf.keras.layers.Input(shape=(num_actions))
        action_out = tf.keras.layers.Dense(32, activation="relu")(action_input)
        # both are passed through seperate layer before concatenating
        concat = tf.keras.layers.Concatenate()([state_out, action_out])
        out = tf.keras.layers.Dense(256, activation="relu")(concat)
        out = tf.keras.layers.Dense(256, activation="relu")(out)
        outputs = tf.keras.layers.Dense(1)(out)
        # output single value for give state-action
        model = tf.keras.Model([state_input,action_input], outputs)
        return model

    """
    policy returns an action sampled from Actor network plus some noise for exploration
    """
    def policy(self, state, noise_object):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state),0)
        sampled_actions = tf.squeeze(self.actor_model(tf_state))
        noise = noise_object()
        sampled_actions = sampled_actions.numpy() + noise
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)]


##########

np.random.seed(123)

def make_video(env, agent, ou_noise):
    env = wrappers.Monitor(env,os.path.join(os.getcwd(),"videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action = agent.policy(obs, ou_noise)
        obs, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
        if done:
            env.reset()
    print("Test Step {} Rewards {}".format(steps, rewards))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ep', type=int, default=100)
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_dev=float(std_dev)*np.ones(1))

    critic_lr = 0.002
    actor_lr = 0.001

    maxEpisode = args.max_ep
    gamma = 0.99
    tau = 0.005

    currTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logDir = 'logs/ddpg' + currTime
    summaryWriter = tf.summary.create_file_writer(logDir)

    env = gym.make("Pendulum-v1")
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    buffer_capacity = 50000
    batch_size = 64

    agent = DDPGAgent(num_states,num_actions,lower_bound,upper_bound,actor_lr,critic_lr,gamma,tau,buffer_capacity,batch_size)

    ep_reward_list = []
    avg_reward_list = []
    for ep in range(maxEpisode):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.policy(state, ou_noise)
            new_state, reward, done, _ = env.step(action)
            agent.buffer.record((state,action,reward,new_state))
            ep_reward += reward
            # learn and update target actor and critic network
            agent.learn()
            agent.update_target(agent.target_actor.variables, agent.actor_model.variables)
            agent.update_target(agent.target_critic.variables, agent.critic_model.variables)

            if done:
                break
            state = new_state

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_reward, step=ep)

        ep_reward_list.append(ep_reward)
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    make_video(env, agent, ou_noise)

    env.close()
