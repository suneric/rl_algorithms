import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

"""
multiple layer perception model
"""
def mlp_model(input_dim, output_dim, hidden_sizes, activation, output_activation):
    input = layers.Input(shape=(input_dim))
    x = layers.Dense(hidden_sizes[0], activation=activation)(input)
    for i in range(1, len(hidden_sizes)):
        x = layers.Dense(hidden_sizes[i], activation=activation)(x)
    output = layers.Dense(output_dim, activation=output_activation)(x)
    return tf.keras.Model(input, output)

"""
Actor-Critic
pi: policy network
q: q-function network
"""
class ActorCritic:
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        self.pi = self.actor_model(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = self.critic_model(obs_dim, act_dim, hidden_sizes, activation)
        print(self.pi.summary())
        print(self.q.summary())

    def actor_model(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        input = layers.Input(shape=(obs_dim))
        x = layers.Dense(hidden_sizes[0],activation=activation)(input)
        for i in range(1, len(hidden_sizes)):
            x = layers.Dense(hidden_sizes[i], activation=activation)(x)
        output = layers.Dense(act_dim, activation='tanh')(x)
        output = output * act_limit
        return tf.keras.Model(input, output)

    def critic_model(self, obs_dim, act_dim, hidden_sizes, activation):
        obs_input = layers.Input(shape=(obs_dim))
        act_input = layers.Input(shape=(act_dim))
        x = layers.Concatenate()([obs_input, act_input])
        for i in range(0, len(hidden_sizes)):
            x = layers.Dense(hidden_sizes[i], activation=activation)(x)
        output = layers.Dense(1, activation='relu')(x)
        model = tf.keras.Model([obs_input,act_input], output)
        return model

    def act(self, obs):
        return self.pi(obs).numpy()
