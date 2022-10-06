# Introduction of RL

## Key Concepts and Terminology
The main characters of RL are the **agent** and the **environment**. The environment is the world that the agent lives in and interacts with. At every step of interaction, the agent sees a (possibly partial) observation of the **state** of the world, and then decides on an **action** to take. The environment changes when the agent acts on it, but may also change on its own. The agent also perceives a **reward** signal from the environment, a number that tells it how good or bad the current world state is. The goal of the agent is to maximize its cumulative reward, called **return**. RL methods are ways that the agent can learn behaviors to achieve its goal.

### States and Observations
A **state** `s` is a complete description of the state of the world. There is no information about the world which is hidden from the state. An **observation** `o` is a partial description of a state, which may omit information. In deep RL, we almost always represent states and observations by a real-valued vector, matrix, or higher-order tensor. For instance, a visual observation could be represented by the RGB matrix of its pixel values; the state of a robot might be represented by its joint angles and velocities. When the agent is able to observe the complete state of the environment, we say that the environment is **fully observed**. When the agent can only see a partial observation, we say that the environment is **partially observed**.

### Action Space
Different environments allow different kinds of actions. The set of all valid actions in a given environment is often called the action space which could be **discrete** where only a finite number of moves are available to the agent or **continuous** where actions are real-valued vectors.

### Policies
A policy is a rule used by an agent to decide what actions to take. It can be **deterministic**, in which case it is usually denoted by $\mu$: $a_t = \mu(s_t)$. Or it may be **stochastic**, in which case it is usually denoted by $\pi$: $a_t~\pi(\dot|s_t)$.
In deep RL, we deal with parameterized policies: whose outputs are computable functions that depend on a set of parameters which can adjust to change the behavior via some optimization algorithm. Such a policy is often denoted by a $\theta$ or $\phi$: $a_t = \mu_{\theta}(s_t)$

#### Deterministic Policies
The output of a deterministic police (for example a MLP network) is determined.

#### Stochastic Policies
The two common kinds of stochastic policies in deep RL are **categorical policies** which can be used in discrete actions space and **diagonal Gussian policies** which are used in continuous action space.
Two key computations are centrally important for using and training stochastic policies:
- sampling actions from the policy
- and computing log likelihoods of particular actions.
**Sampling**, given the probabilities for each action, frameworks like Tensorflow and PyTorch have built-in tools for sampling. (for example tf.distributions.Categorical)
**Log-Likelihood** denote the last layer of probabilities. It is a vector with however many entries as there are actions, so we can treat the actions as indices for the vector. The log likelihood for an action `a` can then be obtained by indexing into the vector.

### Reward and Return
The reward function `R` is critically important in RL. It depends on the current state of the world, the action just taken, and the next state of the world: $r_t = R(s_t,a_t,s_{t+1})$. The goal of the agent is to maximize some notion of cumulative reward of a trajectory (episode or rollout). One kind of return is the **finite-horizon undiscounted return**, which is just the sum of rewards obtained in a fixed windows of steps. Another kind of return is the **infinite-horizon discounted return**, which is the sum of all rewards ever obtained by the agent, but discounted by how far off in the future they're obtained, considering a discount factor $\gamma$ in $(0,1)$.

### Value Functions
It is often useful to know the value of a state, or a state-action pair. By value, we mean the expected return if you start in that state or state-action pair, and then according to a particular policy forever after. Value functions are used, one way or another, in almost every RL algorithm.
- The **On-policy Value Function**, which gives the expected return if you start in state `s` and always actor according to policy $\pi$
- The **On-policy Action-Value Function**, Which gives the expected return if you start in state `s`, take and arbitrary action `a`, and then forever after act according to policy $\pi$
- The **Optimal Value Function**, which gives the expected return if you start in state `s` and always act according to the optimal policy in the environment.
- The **Optimal Action-Value Function**, Which gives the expected return if you start in state `s`, take an arbitrary action `a`, and then forever after act according to the optimal policy in the environment.

### Bellman Equations
All four of the value functions obey special self-consistency equations called Bellman equations: The value of your starting point is the reward you expect to get from being there, plus the value of wherever you land next.

### Advantage Functions
Sometime in RL, we don't need to describe how good an action is in an absolute sense, but only how much better it is than others on average. That is to say, we want to know the relative advantage of that action, we make this concept precise with **advantage function**. Mathematically defined by
$$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$$

## Kinds of RL Algorithms

### model-based or model-free
One of the most important branching points in an RL algorithm is the question of **whether the agent has access to (or learns) a model of the environment**. By a model of the environment, we mean a function which predicts state transitions and rewards. The main upside to having a model is that it allows the agent to plan by thinking ahead, seeing what would happen for a range of possible choices, and explicitly deciding between its options. Agents can then distill the results from planning ahead into a learned policy. The main downside is that a ground-truth model of the environment is usually not available to the agent. Algorithms which use a model are called **model-based** methods, and those that don't are called **model-free**. While model-free methods forego the potential gains in sample efficiency from using a model, they tend to be easier to implement and tune. Model-free method are more popular and have been more extensively developed and tested than model-based methods.

### What to learn
There are two main approaches to representing and training agents with model-free RL:
- **Policy Optimization**, Methods in this family represent a policy explicitly. They optimize the parameter $\theta$ either directly by gradient ascent on performance objective $J(\pi_{\theta})$, or indirectly, by maximizing local approximation of $J(\pi_{\theta})$. This optimization is almost always performed **on-policy**, which means that each update only use data collected while acting according to the most recent version of the policy.
  - A2C/A3C, which performs gradient ascent to directly maximize performance
  - PPO, whose updates indirectly maximize performance, by instead maximizing a surrogate objective function which gives a conservative estimate for how much $J(\pi_{\theta})$ will change as a result of the update.

- **Q-Learning**, Methods in this family learn an approximator $Q_{\theta}(s,a)$ for the optimal action-value function. Typically they use an objective function based on Bellman Equation. This optimization is almost always perform **off-policy**, which means that each update can use data collected at any point during training, regardless of how the agent was choosing to explore the environment when the data was obtained.
  - DQN, a classic which substantially launched the field of deep RL.
  - C51, a variant that learns a distribution over return whose expectation is $Q^*$
- **Trade-offs Between Policy Optimization and Q-Learning**, The primary strength of policy optimization methods is that they are principled, in the sense that you directly optimize for the thing you want. This tends to make them stable and reliable. By contrast, Q-learning methods only indirectly optimize for agent performance, by training Q to satisfy a self-consistency equation.  There are many failure mode for this kind of learning, so it tends to be less stable. But Q-learning methods gain advantage of being substantially more sample efficiency when they do work, because they can reuse data more effective than policy optimization techniques. There exist a range of algorithms that live in between two extremes.
  - DDPG, an algorithm which concurrently learns a deterministic policy and a Q-function by using each to improve the other
  - SAC, a variant which use stochastic policy, entropy regularization, and a few other tricks to stabilize learning and score higher than DDPG on standard benchmarks.
