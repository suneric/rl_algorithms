from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

algo = (
    PPOConfig()
    .environment("Taxi-v3")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .framework("tf2")
    .training(model={"fcnet_hiddens":[64,64]})
    .evaluation(evaluation_num_workers=1)
    .build()
)

for _ in range(10):
    result = algo.train()
    print(pretty_print(result))

algo.evaluate()
