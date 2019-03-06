import numpy as np


class KArmedBanditEnvironment:
    def __init__(self, k, stdev=1, minimum=-5, maximum=5):
        """
        Initializes an environment
        :param k: number of levers in the environment
        :param stdev: default standard deviation for all lever distributions
        :param minimum: minimum allowed reward
        :param maximum: maximum allowed reward
        """
        self.n_arms = k
        self.arms = [Lever(stdev, minimum, maximum) for _ in range(0, self.n_arms)]
        self.plays = 0

    def __best_action(self):
        """
        Private method that returns the lever with highest mean
        """
        return np.argmax([arm.mean for arm in self.arms])

    def get_samples(self, sample_size=10000):
        return [arm.random(sample_size) for arm in self.arms]

    def perform_action(self, k):
        """
        Actually activates a lever
        :param k: selected lever to be activated
        :return: reward given by lever k
        """
        assert (k < self.n_arms)
        reward = self.arms[k].get_sample()
        self.plays += 1
        return reward

    def __str__(self):
        """
        Prints the object metadata
        """
        return str([str(arm) for arm in self.arms])


class Lever:
    def __init__(self, stdev, minimum, maximum):
        """
        Initializes a lever with specific parameters
        :param stdev:   Standard deviation for the distribution
        :param minimum: Minimum accepted value as reward (to prevent outliers)
        :param maximum: Maximum accepted value as reward (to prevent outliers)
        """

        # The mean is randomly selected
        self.mean = np.random.uniform(minimum, maximum)
        self.stdev = stdev
        self.min = minimum
        self.max = maximum

    def random(self, sample_size=None):
        """
        Returns one or more random values from the distribution
        :param sample_size: number of samples to return, default None = 1 sample
        """
        return np.random.normal(self.mean, self.stdev, sample_size)

    def get_sample(self):
        """
        Activates the lever and returns a valid reward
        """
        reward = self.random()
        while reward < self.min or reward > self.max:
            reward = self.random()

        return reward

    def __str__(self):
        """
        Prints the object metadata
        """
        return "mean: {}, stdev:{}".format(self.mean, self.stdev)
