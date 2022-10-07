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

$$\hat{g} = \frac{1}{|\mathbb{D}|}\sum_{\tau \in \mathbb{D}}\sum_{t=0,T}\nabla_{\theta}log\pi_{\theta}(a_t|s_t)R(\tau)$$

where $|\mathbb{D}|$ is the number of trajectories in $\mathbb{D}$ (here, `N`).

This last expression is the simplest version of the computable expression we desired. Assuming that we have represented our policy in a way which allows us to calculate $\nabla_{\theta}log\pi_{\theta}(a|s)$ and if we are able to run the policy in the environment to collect the trajectory dataset, we can compute the policy gradient and take an update step.  

**EGLP Lemma**. Suppose the $P_{\theta}$ us a parameterized probability distribution over a random variable `x`. Then

$$\mathbb{E}_{x \sim P_{\theta}}[\nabla_{\theta}log P_{\theta}(x)] = 0$$

Agents should really only reinforce actions on the basis of their consequences. Rewards obtained before taking an action have no bearing on how good that action was: only rewards that come after. It turns out that this intuition shows up in the math, and we can show that policy gradient can also be expressed by

$$\nabla_{\theta}J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_{t=0,T}\nabla_{\theta}log\pi_{\theta}(a_t|s_t)\sum_{t'=t,T}R(s_{t'},a_{t'},s_{t'+1})]$$

In this form, actions are only reinforced based on rewards obtained after they are taken. We call this form the **reward-to-go** policy gradient, because the sum of rewards after a point in a trajectory.

**Baseline in Policy Gradients**. An immediate consequence of the EGLP lemma is that for any function `b` (called **baseline**) which only depends on state, allows us to add or subtract any number of terms to the form without changing it in expectation

$$\nabla_{\theta}J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_{t=0,T}\nabla_{\theta}log\pi_{\theta}(a_t|s_t)(\sum_{t'=t,T}R(s_{t'},a_{t'},s_{t'+1}) - b(s_t))]$$

The most common choice of a baseline is the on-policy value function $V^{\pi}(s_t)$, this is the average return an agent gets if it starts in state $s_t$ and then acts according to policy $\pi$ for the rest of its life. Empirically, the choice $b(s_t) = V^{\pi}(s_t)$ has the desirable effect of reducing variance in the sample estimate for the policy gradient. This results in faster and more stable policy learning. It is also appealing from a conceptual angle: it encodes the intuition that if an agent gets what it expected, it should "feel" neutral about it. In practice, $V^{\pi}(s_t)$ cannot be computed exactly, so it has to be approximated. This is usually done with a neural network $V_{\phi}(s_t)$, which is updated concurrently with the policy, the simplest method for learning $V_{\phi}$, used in most implementations of policy optimization algorithms (including VPG, TRPO, PPO and A2C) is to minimize a mean-squared-error objective.

**Other Forms of the Policy Gradient**. The policy gradient has a general form

$$\nabla_{\theta}J(\pi_{\theta}) = \mathbb{E}[\sum_{t=0,T}\nabla_{\theta}log\pi_{\theta}(a_t|s_t)\Phi_t]$$

where $\Phi_t$ could be
1. On-Policy Action-Value Function $\Phi_t = Q^{\pi_{\theta}}(s_t,a_t)$.  
2. The Advantage Function $A^{\pi}(s_t,a_t) = Q^{\pi}(s_t,a_t) - V^{\pi}(s_t)$ which describes how much better or worse it is than other actions on average.

**Key Equations of Policy Optimization**. Let $\pi_{\theta}$ denote a policy with parameters $\theta$, and $J(\pi_{\theta})$ denote the expected finite-horizon undiscounted return of the policy. The gradient of $J(\pi_{\theta})$ is

$$\nabla_{\theta}J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_{t=0,T}\nabla_{\theta}log\pi_{\theta}(a_t|s_t)A^{\pi_{\theta}}(s_t,a_t)]$$

where $\tau$ is a trajectory and $A^{\pi_{\theta}}$ is the advantage function for the current policy. The policy gradient algorithm works by updating policy parameters via stochastic gradient ascent on policy performance:

$$\theta_{k+1} = \theta_{k} + \alpha\nabla_{\theta}(\pi_{\theta_k})$$

Policy gradient implementations typically compute advantage function estimates based on the infinite-horizon discounted return, despite otherwise using the finite-horizon undiscounted policy gradient formula.

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

### VPG, NPG, TRPO (Model-Free, On-Policy, Discrete or Continuous Action Space)
As promised, we can not only choose the target $\Phi_t$, but also have some freedom when it comes to the vector $g_t = \nabla log\pi_{\theta}(a_t|s_t)$ in whose direction we update the parameter $\theta$.

The simplest Policy Optimization algorithm is **Vanilla Policy Gradient** (VPG), which use

$$g_t = \nabla log\pi_{\theta}(a_t|s_t)$$

But this simple method has its drawback: gradient descent leads to small changes in the parameter $\theta$, but it doesn't make any guarantees about the changes in the policy $\pi$ itself. If the policy is very sensitive to the parameter around some value $\theta_0$, then taking a gradient step from there might change the policy a lot and actually make it worse. To avoid that, we'll need to use a small learning rate, which shows down convergence.

<p align="center">
<img src="https://github.com/suneric/rl_algorithms/blob/main/references/vpg_algo.svg" width=80% height=80%>
</p>

[TRPO (Trust Region Policy Optimization): Schulman et al, 2015](https://arxiv.org/abs/1502.05477)

The solution is to use the **Natural Policy Gradient** (NPG) instead of the usual gradient. Instead of limiting the size of the step in parameter space, it directly limits the change of the policy at each step. Natural gradients are a general method for finding optimal probability distribution, not specific to RL, but NPG is probably their most well-know application. Computationally, **the natural gradient is just the normal gradient multiplied by the inverse Fisher matrix $F^{-1}$ of the policy**:

$$g_t = F^{-1}\nabla log\pi_{theta}(a_t|s_t)$$.

A third option is **Trust-Region Policy Optimization** (TRPO). The motivation is similar to that of NPG: limit how much the policy changes (in terms of the KL divergence). But it takes that idea further and actually guarantees an upper bound on how much the policy will change. Use the same update vector as NPG with a learning rate that adapts at each step: $g_t = F^{-1}$\nabla log\pi_{theta}(a_t|s_t)$ and the adaptive learning rate $\alpha = \beta^j\sqrt{\frac{2\delta}{\tilde{g}F^{-1}\tilde{g}}}$ where $\tilde{g} = \Phi_tg_t$, $\beta \in (0,1)$ and $\delta$ are hyper-parameters and $j \in \mathbb{N}_0$ is chosen minimally such that a constraint on the KL divergence between old and new policy is satisfied.

<p align="center">
<img src="https://github.com/suneric/rl_algorithms/blob/main/references/trpo_algo.svg" width=80% height=80%>
</p>

### PPO

<p align="center">
<img src="https://github.com/suneric/rl_algorithms/blob/main/references/ppo_algo.svg" width=80% height=80%>
</p>

[PPO (Proximal Policy Optimization): Schulman et al, 2017](https://arxiv.org/abs/1707.06347)

### DDPG

<p align="center">
<img src="https://github.com/suneric/rl_algorithms/blob/main/references/ddpg_algo.svg" width=80% height=80%>
</p>
[DDPG (Deep Deterministic Policy Gradient): Lillicrap et al, 2015](https://arxiv.org/abs/1509.02971

### TD3

<p align="center">
<img src="https://github.com/suneric/rl_algorithms/blob/main/references/td3_algo.svg" width=80% height=80%>
</p>

[TD3 (Twin Delayed DDPG): Fujimoto et al, 2018](https://arxiv.org/abs/1802.09477)

### SAC

<p align="center">
<img src="https://github.com/suneric/rl_algorithms/blob/main/references/sac_algo.svg" width=80% height=80%>
</p>

[SAC (Soft Actor-Critic): Haarnoja et al, 2018](https://arxiv.org/abs/1801.01290)

## References
- [Open AI Spinning up, Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- [Tensorflow, Introduction to RL and Deep Q Networks](https://www.tensorflow.org/agents/tutorials/0_intro_rl)
- [Latex Math Symbols](https://kapeli.com/cheat_sheets/LaTeX_Math_Symbols.docset/Contents/Resources/Documents/index)
