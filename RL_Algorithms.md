# RL Algorithms

<p align="center">
<img src="https://github.com/suneric/rl_algorithms/blob/main/references/rl_algorithms.svg" width=80% height=80%>
</p>

## Model-Free RL
One of the most important branching points in an RL algorithm is the question of **whether the agent has access to (or learns) a model (which predicts state transitions and rewards) of the environment**. The **main upside** of having a model is that it allows the agent to plan by thinking ahead, seeing what would happen for a range of possible choices, and explicitly deciding between its options. Agents can then distill the results from planning ahead into a learned policy. The **main downside** is that a ground-truth model of the environment is usually not available to the agent. If an agent wants to use a model in this case, it has to learn the model purely from experience, which creates several challenges. The biggest challenge is that bias in the model can be exploited by the agent, resulting in an agent which performs well with respect to the learned model, but behaves sub-optimally (or super terribly) in the real environment. **Model-learning is fundamentally hard, so even intense effort-being willing to throw lots of time and compute at it-can fail to pay off. While model-free methods forgo the potential gains in sample efficiency from using a model, they tend to be easier to implement and tune**. Thus, model-free methods are more popular and have been more extensively developed and tested than model-based methods.

### Q-Learning
Q-Learning is based on the notion of Q-function (a.k.a the state-action value function) of a policy $\pi$, $Q^{\pi}(s,a)$, which measures the expected return or discounted sum of rewards obtained from state `s` by taking action $a$ first and following policy $\pi$ thereafter.The **optimal** Q-function $Q^{\*}(s,a)$ is defined as the maximum return that can obtained starting from observation `s`, taking action `a` and following the optimal policy thereafter. The optimal Q-function obeys the following **Bellman optimality** equation $$Q^{\*}(s,a) = \mathbb{E}[r + \gamma\max_{a'}Q^{\*}(s',a')]$$ This means that the maximum return from stats `s` and action `a` is the sum of the immediate reward `r` and the return (discounted by $\gamma$) obtained by following the optimal policy thereafter until th end of the episode. The expectation is computed both over the distribution of immediate reward `r` and possible next state `s'`. The basic idea behind Q-Learning is to use the Bellman optimality equation as an iterative update $$Q_{i+1}(s,a) \gets \mathbb{E}[r + \gamma\max_{a'}Q_i(s',a')]$$ and it can be shown that this converges to the optimal Q-function.

### Policy Optimization
Policy optimization in general means that we have a parameterized family of policies $\pi_{\theta}(a|s)$ and want to maximize the expected return with respect to the parameters $\theta$: $arg\max_{\theta} J(\theta)$ where `J` is the expected return

$$J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]$$

For the purpose of this derivation, we take $R(\tau)$ to give the **finite-horizon undiscounted return**, but the derivation for the infinite-horizon discounted return setting is almost identical. We would like to optimize the policy by gradient ascent, e.g.

$$\theta_{k+1} = \theta_k + \alpha \nabla_{\theta}J(\pi_{\theta})|_{\theta_k}$$

The gradient of policy performance, $\nabla_{\theta}J(\pi_{\theta})$, is called the **policy gradient**. To actually use this algorithm, we need an expression for the policy gradient which can numerically compute. This involves two steps: 1) deriving the analytical gradient of policy performance, which turns out to have the form of an expected value and then 2) forming a sample estimate of the expected value, which can be computed with data from a finite number of agent-environment interaction steps.
1. **probability of a Trajectory**. The probability of a trajectory $\tau = (s_0,a_0,...,s_{T+1})$ given that actions come from $\pi_{\theta}$ is
$$P(\tau|\theta) = \rho_0(s_0)\prod_{t=0,T}P(s_{t+1}|s_t,a_t)\pi_{\theta}(a_t|s_t)$$
2. **The log-Derivative Trick**. The log-derivative trick is based on a simple rule from calculus: the derivative of `log x` with respect to `x` is `1/x`. When rearranged and combined with chain rule, we get:
$$\nabla_{\theta}P(\tau|\theta) = P(\tau|\theta)\nabla_{\theta} log P(\tau|\theta)$$
3. **Log-probability of a Trajectory**. The log-prob of a trajectory is just
$$log P(\tau|\theta) = log \rho_0(s_0) + \sum_{t=0,T}(log P(s_{t+1}|s_t,a_t) + log \pi_{\theta}(a_t|s_t))$$
4. **Gradients of Environment Functions**. The environment has no dependence on $\theta$, so gradients of $\rho_0(s_0)$, $P(s_{t+1}|s_t,a_t)$ and $R(\tau)$ are zero.
5. **Grad-Log-Prob of a Trajectory**. The gradient of the log-prob of a trajectory is thus
$$\nabla_{\theta}log P(\tau|\theta) = \sum_{t=0,T} \nabla_{\theta}log \pi_{\theta}(a_t|s_t)$$
This is an expectation, which means that we can estimate it with a sample mean. If we collect a set of trajectory $\mathbb{D} = {\tau_i}_{i=1,...,N}$
where each trajectory is obtained by letting the agent act in the environment using the policy $\pi_{\theta}$, the policy gradient can be estimated with

$$\hat{g} = {1 /over |\mathbb{D}|}\sum_{\tau \in \mathbb{D}}\sum_{t=0,T}\nabla_{\theta}log\pi_{\theta}(a_t|s_t)R(\tau)$$

where $|\mathbb{D}|$ is the number of trajectories in $\mathbb{D}$ (here, `N`).

This last expression is the simplest version of the computable expression we desired. Assuming that we have represented our policy in a way which allows us to calculate $\nabla_{\theta}log\pi_{\theta}(a|s)$ and if we are able to run the policy in the environment to collect the trajectory dataset, we can compute the policy gradient and take an update step.  

## Popular Algorithms
### DQN (Model-Free, Off-Policy, Discrete Action Space)
For most problems, it is impractical to represent the Q-function as a table containing values for each combination of `s` and `a`, Instead, we train a **function approximator**, such as a neural network with parameter $\theta$, to estimate the Q-values. i.e $Q(s,a;\theta) \approx Q^{\*}(s,a)$. This can be done by minimizing the following loss at each step `i`:

$$\mathbb{L}(\theta_i) = \mathbb{E}_{s,a,r,s' \sim \rho(.)}[(y_i - Q(s,a;\theta_i))^2]$$

where

$$y_i = r + \gamma\max_{a'}Q(s',a';\theta_{i-1})$$

Here $y_i$ is called the TD(temporal difference) target, and $y_i - Q$ is called the TD error. $\rho$ represents the behavior distribution, the distribution over transition `{s,a,r,s'}` collected from the environment. **Note that the parameters from the previous $\theta_{i-1}$ are fixed and not updated**. In practice we use a snapshot of the network parameters from a few iterations ago instead of the last iteration. This copy is called the **target network**.  

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
- [Latex Math Symbols](https://kapeli.com/cheat_sheets/LaTeX_Math_Symbols.docset/Contents/Resources/Documents/index)
