import numpy as np
from src.bandit import Bandit


class KArmedBanditEnvironment:
    def __init__(self, k, n_bandits=1, stdev=1, minimum=-5, maximum=5):
        """
        Initializes an environment
        :param k: number of levers in the environment
        :param stdev: default standard deviation for all lever distributions
        :param minimum: minimum allowed reward
        :param maximum: maximum allowed reward
        """
        self.n_arms = k
        self.n_bandits = n_bandits
        self.bandits = [Bandit(k, stdev, minimum, maximum) for _ in range(0, self.n_bandits)]
        self.plays = 0
        self.latest_state = self.get_next_state()

    def get_next_state(self):
        return np.random.randint(0, self.n_bandits)

    def perform_action(self, k):
        """
        Actually activates a lever
        :param k: selected lever to be activated
        :return: reward given by lever k
        """
        bandit = self.bandits[self.latest_state]
        reward = bandit.perform_action(k)
        self.latest_state = self.get_next_state()

        self.plays += 1
        return reward, self.latest_state

    def __str__(self):
        """
        Prints the object metadata
        """
        return str([str(bandit) for bandit in self.bandits])
