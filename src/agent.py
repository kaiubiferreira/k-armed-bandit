import numpy as np


class KArmedBanditAgent:
    def __init__(self, k, epsilon=0.1):
        """
        Initializes an agent
        :param k: Number of levers available for the agent
        :param epsilon: Parameter to control exploration and exploitation
        """

        self.n_arms = k
        self.epsilon = epsilon
        self.reward_sum = np.zeros(self.n_arms)
        self.n_plays = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.cumulative_reward = 0

    def get_action(self):
        """
        Uses the computed rewards averages as a value function
        and the epsilon value to decide for the next action
        :return: The agent's next action
        """
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.randint(0, self.n_arms)
        else:
            action = np.argmax(self.values)

        return action

    def set_reward(self, action, reward):
        """
        Updates the value function
        :param action: The agent's previously selected action
        :param reward: The reward received for selection that action
        """
        self.reward_sum[action] += reward
        self.n_plays[action] += 1
        self.values[action] = self.reward_sum[action] / self.n_plays[action]
        self.cumulative_reward += reward