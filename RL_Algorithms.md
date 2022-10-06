# RL Algorithms

<img src="https://github.com/suneric/rl_algorithms/tree/main/references/rl_algorithms.svg" width=50% height=50%>

## Model-Free RL
One of the most important branching points in an RL algorithm is the question of *whether the agent has access to (or learns) a model (which predicts state transitions and rewards) of the environment*. The *main upside* of having a model is that it allows the agent to plan by thinking ahead, seeing what would happen for a range of possible choices, and explicitly deciding between its options. Agents can then distill the results from planning ahead into a learned policy. The *main downside* is that a ground-truth model of the environment is usually not available to the agent. If an agent wants to use a model in this case, it has to learn the model purely from experience, which creates several challenges. The biggest challenge is that bias in the model can be exploited by the agent, resulting in an agent which performs well with respect to the learned model, but behaves sub-optimally (or super terribly) in the real environment. *Model-learning is fundamentally hard, so even intense effort-being willing to throw lots of time and compute at it-can fail to pay off. While model-free methods forgo the potential gains in sample efficiency from using a model, they tend to be easier to implement and tune*. Thus, model-free methods are more popular and have been more extensively developed and tested than model-based methods.

### Q-Learning
Q-Learning is based on the notion of Q-function (a.k.a the state-action value function) of a policy $\pi$, $Q^{\pi}(s,a)$, which measures the expected return or discounted sum of rewards obtained from state $s$ by taking action $a$ first and following policy $\pi$ thereafter.The *optimal* Q-function $Q^*(s,a)$ is defined as the maximum return that can obtained starting from observation $s$, taking action $a$ and following the optimal policy thereafter. The optimal Q-function obeys the following Bellman optimality equation:
$$Q^*(s,a) = \mathbb{E}[r + \gamma\max_{a'}Q^*(s',a')]$$

### Policy OPtimization

## Popular Algorithms
### DQN

[DQN (Deep Q-Network), Mhih et al, 2013](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

### VPG

### PPO

### DDPG

### TD3

### SAC

## References
- [Open AI Spinning up, Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- [Tensorflow, Introduction to RL and Deep Q Networks](https://www.tensorflow.org/agents/tutorials/0_intro_rl)
