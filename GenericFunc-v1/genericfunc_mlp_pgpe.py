"""
-----------------------
Ordinary Function Optimizer
-----------------------

Author: Dimitri Haas
Version: 1.0
Date: 14.05.23

This script is designed to optimize ordinary functions through reinforcement learning, 
specifically utilizing a variant of the Policy Gradients with Parameter-based Exploration (PGPE) algorithm. 
Originally proposed by Frank Sehnke, the variant used here adapts the SyS-PGPE algorithm for a broader 
application beyond specific simulations, allowing for the optimization of any given mathematical or 
geometric function.

The core of the optimizer is a Multi-Layer Perceptron (MLP) with a linear output layer, 
modified to work with arbitrary functions by evaluating their outputs as rewards within a 
custom environment. This environment, `FuncEnv`, is tailored to handle ordinary functions, 
taking actions that correspond to input parameters of the function being optimized.

The flexibility of this approach lies in its application to a wide variety of optimization problems, 
ranging from simple analytical functions to complex, multidimensional landscapes requiring sophisticated 
exploration strategies.

Happy Optimizing!
"""

import numpy as np
import csv
from tqdm import trange


class MLP:
    def __init__(self, topo):
        self.layers = []
        for i in range(len(topo) - 1):
            W = np.random.randn(topo[i + 1], topo[i]) / np.sqrt(topo[i]) * (5 / 3)
            b = np.random.randn(topo[i + 1]) / np.sqrt(topo[i])
            self.layers.append((W, b))

    def linear(self, x, W, b):
        return W @ x + b

    def forward(self, x):
        a = x
        for W, b in self.layers[:-1]:
            a = np.tanh(self.linear(a, W, b))
        W, b = self.layers[-1]
        return self.linear(a, W, b)

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


class FuncEnv:
    def __init__(self, function, consts=None, bounds=None, objective="max"):
        super(FuncEnv, self).__init__()
        self.function = function
        self.consts = consts
        self.bounds = bounds
        self.objective = objective

    def step(self, action):
        cost = self.function(*action)
        const_value = self.consts(*action) if self.consts is not None else 0

        if self.objective == "max":
            reward = cost - 100 * abs(const_value) ** 2
        else:  # Assuming the objective is 'min'
            reward = -cost + 100 * abs(const_value) ** 2

        return np.array(action), reward, True, False, {}

    def reset(self):
        return np.array([1.0])  # Dummy


class PGPE:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.mu = policy.get_weights()
        self.p_count = len(self.mu)
        self.sigma = np.ones(self.p_count) * 2
        self.baseline = -2e6
        self.best = -np.inf
        self.learn_rate = 0.04

    def run_episode(self):
        episodic_reward = 0
        states = self.env.reset()

        while True:
            action_prob = self.policy.forward(states)
            states, reward, terminated, truncated, _ = self.env.step(action_prob)
            episodic_reward += reward
            if terminated or truncated:
                break
        return episodic_reward, action_prob

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
        self.baseline = 0.95 * self.baseline + 0.025 * sum(fit)

        self.mu += self.learn_rate * mu_grad * perturb

        if std_grad > 0.0:  # updates only if that leads to better results
            expl = perturb**2 - self.sigma**2  # extend/shrink the search space
            self.sigma += self.learn_rate / 2 * std_grad * expl / self.sigma

    def learn(self, iterations):
        data = []
        fit = np.zeros(2)
        fmt = "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} | {unit}"
        for _ in (pbar := trange(iterations, bar_format=fmt)):
            perturb = np.random.normal(0, self.sigma, self.p_count)
            self.policy.set_weights(self.mu + perturb)
            fit[0], p_action = self.run_episode()
            self.policy.set_weights(self.mu - perturb)
            fit[1], n_action = self.run_episode()
            m_action = (p_action + n_action) / 2
            pbar.unit = f"x={m_action[0]:>4.1f}, y={m_action[1]:>4.1f}"
            data.append((*m_action, sum(fit) / 2))

            self.update(perturb, fit)
        # Saving data to CSV
        with open("GenericFunc-v1/results.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Y", "Reward"])  # Header
            for row in data:
                writer.writerow(row)


if __name__ == "__main__":
    env = FuncEnv(
        function=lambda x, y: -(x**2 + (y - 2) ** 2),
        consts=lambda x, y: x**2 - y**2 - 1,
        objective="max",
    )
    policy = MLP([1, 16, 2])
    agent = PGPE(env, policy)
    agent.learn(iterations=25_000)
    np.savetxt("GenericFunc-v1/weights/mlp_pgpe.txt", agent.policy.get_weights())
