# RL Algorithms

<img src="https://github.com/suneric/rl_algorithms/blob/main/references/rl_algorithms.svg" width=50% height=50%>

## Model-Free RL
One of the most important branching points in an RL algorithm is the question of **whether the agent has access to (or learns) a model (which predicts state transitions and rewards) of the environment**. The **main upside** of having a model is that it allows the agent to plan by thinking ahead, seeing what would happen for a range of possible choices, and explicitly deciding between its options. Agents can then distill the results from planning ahead into a learned policy. The **main downside** is that a ground-truth model of the environment is usually not available to the agent. If an agent wants to use a model in this case, it has to learn the model purely from experience, which creates several challenges. The biggest challenge is that bias in the model can be exploited by the agent, resulting in an agent which performs well with respect to the learned model, but behaves sub-optimally (or super terribly) in the real environment. **Model-learning is fundamentally hard, so even intense effort-being willing to throw lots of time and compute at it-can fail to pay off. While model-free methods forgo the potential gains in sample efficiency from using a model, they tend to be easier to implement and tune**. Thus, model-free methods are more popular and have been more extensively developed and tested than model-based methods.

### Q-Learning
Q-Learning is based on the notion of Q-function (a.k.a the state-action value function) of a policy $\pi$, $Q^{\pi}(s,a)$, which measures the expected return or discounted sum of rewards obtained from state `s` by taking action $a$ first and following policy $\pi$ thereafter.The **optimal** Q-function $Q^*(s,a)$ is defined as the maximum return that can obtained starting from observation `s`, taking action `a` and following the optimal policy thereafter. The optimal Q-function obeys the following Bellman optimality equation $$Q^{\*}(s,a) = \mathbb{E}[r + \gamma\max_{a'}Q^{\*}(s',a')]$$ This means that the maximum return from stats `s` and action `a` is the sum of the immediate reward `r` and the return (discounted by $\gamma$) obtained by following the optimal policy thereafter until th end of the episode. The expectation is computed both over the distribution of immediate reward `r` and possible next state `s'`. The basic idea behind Q-Learning is to use the Bellman optimality equation as an iterative update $$Q_{i+1}(s,a) \from \mathbb{E}[r + \gamma\max_{a'}Q_i(s',a')]$$ and it can be shown that this converges to the optimal Q-function.

### Policy Optimization

## Popular Algorithms
### DQN (Model-Free, Off-Policy)
For most problems, it is impractical to represent the Q-function as a table containing values for each combination of `s` and `a`, Instead, we train a **function approximator**, such as a neural network with parameter $\theta$, to estimate the Q-values. i.e $Q(s,a;\theta) \approxeq Q^{\*}(s,a)$. This can be done by minimizing the following loss at each step `i`: $$\mathbb{L}_i(\theta_i) = \mathbb{E}_{s,a,r,s' \sim \rou(\.)}[(y_i - Q(s,a;\theta_i))^2]$$ where $$y_i = r + \gamma\max_{a'}Q(s',a';\theta_{i-1})$$ Here,$y_i$ is called the TD(temporal difference) target, and $y_i - Q$ is called the TD error. $\rou$ represents the behavior distribution, the distribution over transition `{s,a,r,s'}` collected from the environment. **Note that the parameters from the previous $\theta_{i-1}$ are fixed and not updated**. In practice we use a snapshot of the network parameters from a few iterations ago instead of the last iteration. This copy is called the **target network**.  

Q-Learning algorithm leans about the greedy policy $a = \max_aQ(s,a;\theta)$ while using a different behavior policy for acting in the environment/collecting data. This behavior policy is usually an $\epsilon$-greedy policy that selects the greedy action with probability $1-\epsilon$ and a random action with probability $\epsilon$ to ensure good coverage of the state-action space.

To avoid computing the full experience in the DQN loss, we can minimize it using stochastic gradient descent. If the loss is computed using just the last transition `{s,a,r,s'}`, this reduces to standard Q-Learning. DQN introduced a technique called Experience Replay to make the network updates more stable. At each time step of data collection, the transitions are added to a circular buffer called the **replay buffer**. Then during training, instead of using just the latest transition to compute the loss and its gradient, we compute them using a mini-batch of transitions sampled from the replay buffer. **This has two advantages: better data efficiency by reusing each transition in many updates, and better stability using uncorrelated transitions in a batch**. 

[DQN (Deep Q-Network), Mhih et al, 2013](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

### VPG

### PPO

### DDPG

### TD3

### SAC

## References
- [Open AI Spinning up, Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- [Tensorflow, Introduction to RL and Deep Q Networks](https://www.tensorflow.org/agents/tutorials/0_intro_rl)
- [Mathematics in R Markdown](https://rpruim.github.io/s341/S19/from-class/MathinRmd.html)
