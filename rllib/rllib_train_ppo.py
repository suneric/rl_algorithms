"""

"""
import ray
from ray import tune
from ray.rllib import agents
import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

ray.init()

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 1
config["num_workers"] = 2
algo = ppo.PPO(config=config, env='LunarLander-v2')
for s in range(1000):
    result = algo.train()
    print(pretty_print(result))

# algo.evaluate()
