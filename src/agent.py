import numpy as np


class KArmedBanditAgent:
    def __init__(self, k, n_bandits=1, epsilon=0.1):
        """
        Initializes an agent
        :param k: Number of levers available for the agent
        :param epsilon: Parameter to control exploration and exploitation
        """

        self.n_arms = k
        self.n_bandits = n_bandits
        self.epsilon = epsilon
        self.reward_sum = {bandit: np.zeros(self.n_arms) for bandit in range(0, n_bandits)}
        self.n_plays = {bandit: np.zeros(self.n_arms) for bandit in range(0, n_bandits)}
        self.values = {bandit: np.zeros(self.n_arms) for bandit in range(0, n_bandits)}
        self.cumulative_reward = {bandit: 0 for bandit in range(0, n_bandits)}

    def get_action(self, state):
        """
        Uses the computed rewards averages as a value function
        and the epsilon value to decide for the next action
        :return: The agent's next action
        """
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.randint(0, self.n_arms)
        else:
            action = np.argmax(self.values[state])

        return action

    def set_reward(self, current_state, action, reward):
        """
        Updates the value function
        :param current_state: The bandit select
        :param action: The agent's previously selected action
        :param reward: The reward received for selection that action
        """
        self.reward_sum[current_state][action] += reward
        self.n_plays[current_state][action] += 1
        self.values[current_state][action] = self.reward_sum[current_state][action] / self.n_plays[current_state][action]
        self.cumulative_reward[current_state] += reward
