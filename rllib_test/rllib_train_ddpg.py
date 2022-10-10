"""

"""
import ray
from ray import tune
import ray.rllib.algorithms.ddpg as ddpg
from ray.tune.logger import pretty_print

ray.init()

"""
training
"""
config = ddpg.DDPGConfig().to_dict()
config['env'] = 'LunarLanderContinuous-v2'
config['twin_q'] = False # true for TD3
config['target_noise'] = 0.2
config['target_noise_clip'] = 0.5
config['actor_hiddens'] = [256,256,64]
config['actor_hidden_activation'] = 'relu'
config['critic_hiddens'] = [256,256,64]
config['critic_hidden_activation'] = 'relu'
config['actor_lr'] = 1e-4
config['critic_lr'] = 1e-3
config['gamma'] = 0.99
config['tau'] = 0.005
config['train_batch_size'] = 64
# config['exploration_config']
# config['replay_buffer_config']
config['num_gpus'] = 1
config['num_workers'] = 1

# algo = ddpg.DDPG(config = config)
# print(algo.get_policy().model.policy_model.summary())
# # print(algo.get_policy().model.q_model.summary())
#
# for _ in range(10):
#     result = algo.train()
#     print("iteration {}, eps {}, mean reward {}, total steps {}".format(
#         result['training_iteration'],
#         result['episodes_this_iter'],
#         result['episode_reward_mean'],
#         result['timesteps_total']
#     ))


"""
Tuning
"""

from ray import tune

analysis = tune.run(
    'DDPG',
    num_samples=4,
    stop = {"training_iteration": 100},
    config = {
        "env": 'LunarLanderContinuous-v2',
        "num_workers": 3,
        "critic_lr": tune.grid_search([1e-2,1e-3,1e-4])
    }
)
print(analysis.get_best_config(metric="episode_reward_mean"))
