import gym
from gym import envs
print(envs.registry.all())

env = gym.make('CartPole-v1')
env.reset()
for _ in range(1000):
    env.render()
    obs, r, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
env.close()
