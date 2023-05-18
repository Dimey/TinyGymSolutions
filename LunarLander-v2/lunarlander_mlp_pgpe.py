"""
-----------------------
Lunar Lander Agent Trainer
-----------------------

Author: Dimitri Haas
Version: 1.0
Date: 14.05.23

This script trains a Lunar Lander agent using a variant of the Policy Gradients with Parameter-based Exploration (PGPE) algorithm. 
The variant used is called SyS-PGPE, originally proposed by Frank Sehnke.

The agent's policy is a Multi-Layer Perceptron (MLP) with a softmax output layer.

The trainer can be used in two modes by setting the train variable:
- Training mode: The agent is trained for a specified number of iterations. The resulting weights are saved to a text file.
- Evaluation mode: The agent's performance is evaluated over a number of episodes, using pre-trained weights loaded from a text file.

Happy Landing!

"""

import numpy as np
import gymnasium as gym
from tqdm import tqdm, trange


class MLP:
    def __init__(self, topo):
        self.layers = []
        for i in range(len(topo) - 1):
            W = np.random.randn(topo[i + 1], topo[i]) / np.sqrt(topo[i])
            b = np.random.randn(topo[i + 1]) / np.sqrt(topo[i])
            self.layers.append((W, b))

    def softmax(self, x):
        norm = np.exp(x - np.max(x))
        return norm / norm.sum()

    def linear(self, x, W, b):
        return W @ x + b

    def forward(self, x):
        a = x
        for W, b in self.layers[:-1]:
            a = np.tanh(self.linear(a, W, b))
        W, b = self.layers[-1]
        return self.softmax(self.linear(a, W, b))

    def set_weights(self, params):
        i = 0
        for j in range(len(self.layers)):
            W, b = self.layers[j]
            W = params[i : i + W.size].reshape(W.shape)
            i += W.size
            b = params[i : i + b.size].reshape(b.shape)
            i += b.size
            self.layers[j] = (W, b)

    def get_weights(self):
        weights = []
        for W, b in self.layers:
            weights.append(W.flatten())
            weights.append(b.flatten())
        return np.concatenate(weights)


class PGPE:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.mu = policy.get_weights()
        self.p_count = len(self.mu)
        self.sigma = np.ones(self.p_count) * 2
        self.baseline = -250.0
        self.best = -np.inf
        self.learn_rate = 0.2

        # State stats were obtained by a previous run with random actions:
        self.s_mean = np.array([0.01, 0.96, 0.03, -0.67, -0.01, 0.00])
        self.s_std = np.array([0.30, 0.47, 0.62, 0.48, 0.53, 0.54])

    def norm_states(self, s):  # only normalizing non-binary states
        return np.concatenate(((s[:6] - self.s_mean) / self.s_std, s[6:]))

    def run_episode(self, seed=None):
        episodic_reward = 0
        states = self.env.reset(seed=seed)[0]

        while True:
            action_prob = self.policy.forward(self.norm_states(states))
            action = np.random.choice(len(action_prob), p=action_prob)
            states, reward, terminated, truncated, _ = self.env.step(action)
            episodic_reward += reward
            if terminated or truncated:
                break
        return episodic_reward

    def update(self, perturb, fit):
        reward = max(fit)

        if reward > self.best:
            self.best = reward

        if fit[0] != fit[1]:
            mu_grad = (fit[0] - fit[1]) / (2 * self.best - fit[0] - fit[1])
        else:
            mu_grad = 0.0
        std_grad = (reward - self.baseline) / (self.best - self.baseline)
        std_grad = np.clip(std_grad, -1.0, 1.0)
        self.baseline = 0.9 * self.baseline + 0.05 * sum(fit)

        self.mu += self.learn_rate * mu_grad * perturb

        if std_grad > 0.0:  # updates only if that leads to better results
            expl = perturb**2 - self.sigma**2  # extend/shrink the search space
            self.sigma += self.learn_rate / 2 * std_grad * expl / self.sigma

    def learn(self, iterations):
        fit = np.zeros(2)
        fmt = "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} | Baseline: {unit}"
        for i in (pbar := trange(iterations, bar_format=fmt)):
            perturb = np.random.normal(0, self.sigma, self.p_count)
            seed = np.random.randint(10_000)
            self.policy.set_weights(self.mu + perturb)
            fit[0] = self.run_episode(seed)
            self.policy.set_weights(self.mu - perturb)
            fit[1] = self.run_episode(seed)
            pbar.unit = f"{self.baseline:>6.1f}"

            self.update(perturb, fit)


if __name__ == "__main__":
    train = True
    env = gym.make("LunarLander-v2", render_mode="rgb_array" if train else "human")
    obs_space = env.observation_space.shape[0]  # type: ignore
    action_space = env.action_space.n  # type: ignore
    policy = MLP([obs_space, 16, action_space])
    agent = PGPE(env, policy)

    if train:
        agent.learn(iterations=3_000)
        np.savetxt("LunarLander-v2/weights/mlp_pgpe.txt", agent.policy.get_weights())
    else:
        agent.policy.set_weights(np.loadtxt("LunarLander-v2/weights/mlp_pgpe.txt"))
        for _ in range(100):
            print(f"Collected Reward: {agent.run_episode():.1f}")
