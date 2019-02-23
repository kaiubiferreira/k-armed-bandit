import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class RandomDistribution:
    def __init__(self, mean=0, stdev=1):
        self.stdev = stdev
        self.mean = np.random.normal(loc=mean, scale=self.stdev)

    def get_sample(self, sample_size=None):
        return np.random.normal(self.mean, self.stdev, sample_size)

    def metadata(self):
        return "mean: {}, stdev:{}".format(self.mean, self.stdev)


class KArmedBandit:
    def __init__(self, k):
        self.n_arms = k
        self.arms = [RandomDistribution() for _ in range(0, self.n_arms)]
        self.plays = 0
        self.palette = sns.color_palette("muted", n_colors=k)

    def best_action(self):
        return np.argmax([arm.mean for arm in self.arms])

    def get_samples(self, sample_size):
        return [arm.get_sample(sample_size) for arm in self.arms]

    def perform_action(self, k):
        assert (k < self.n_arms)
        reward = self.arms[k].get_sample()
        self.plays += 1
        return reward

    def metadata(self):
        return [arm.metadata() for arm in self.arms]


class KArmedBanditLearner(KArmedBandit):
    def __init__(self, k, epsilon):
        super().__init__(k)

        self.total_reward = 0.0
        self.k_armed = KArmedBandit(k)
        self.values = np.zeros(k)
        self.reward_sum = np.zeros(k)
        self.n_plays = np.zeros(k)
        self.epsilon = epsilon
        columns = list(range(0, self.n_arms + 1))
        self.history = pd.DataFrame(columns=columns)
        self.results = pd.DataFrame(columns=['step', 'current', 'best', 'proportion'])
        self.cummulative_reward = 0
        self.optimum_reward = 0

    def get_action(self):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.randint(0, self.n_arms)
        else:
            action = np.argmax(self.values)

        return action

    def run(self):
        for i in range(1, 500):
            action = self.get_action()
            reward = self.perform_action(action)
            self.n_plays[action] += 1
            self.reward_sum[action] += reward
            self.values[action] = self.reward_sum[action] / self.n_plays[action]
            cols = np.append(self.n_plays, i).reshape(1, self.n_arms + 1)
            self.history = self.history.append(pd.DataFrame(cols))

            best_action = self.best_action()
            best_reward = self.perform_action(best_action)

            self.cummulative_reward += reward
            self.optimum_reward += best_reward
            self.results = self.results.append(pd.DataFrame([[i, self.cummulative_reward, self.optimum_reward,
                                                              self.cummulative_reward / self.optimum_reward]],
                                                            columns=self.results.columns))

        self.history = self.history.reset_index(drop=True)
        self.history = pd.melt(self.history, id_vars=self.n_arms, value_vars=list(range(0, agent.n_arms)))

    def plot(self):
        fig, axs = plt.subplots(nrows=3)
        sns.violinplot(data=self.get_samples(10000), ax=axs[0], palette=self.palette)
        sns.barplot(x=list(range(0, self.n_arms)), y=self.n_plays, ax=axs[1], palette=self.palette)
        sns.lineplot(x=self.n_arms, y="value", hue="variable", data=self.history, ax=axs[2], palette=self.palette)
        plt.show()

    def plot_result(self):
        sns.lineplot(x=self.results.step, y=self.results.proportion)
        plt.show()


total_results = pd.DataFrame(columns=['run', 'epsilon', 'current', 'best', 'proportion'])

for epsilon in [0.01, 0.1, 0.0]:
    for i in range(1, 2000):
        print(i)
        agent = KArmedBanditLearner(10, epsilon=0.1)
        agent.run()
        # agent.plot()
        # agent.plot_result()
        current_results = agent.results.reset_index(drop=True)
        current_results['run'] = i
        current_results['epsilon'] = epsilon
        total_results = total_results.append(current_results, sort=True)

total_results.index.name = 'ix'
agg_results = total_results.groupby(['ix', 'epsilon'])['proportion'].agg('mean')\
    .reset_index()\
    .rename(columns={'mean': 'reward'})

pallete = sns.color_palette("muted", n_colors=3)
agg_results.to_pickle("agg_results.pickle")
sns.lineplot(x="ix", y="proportion", hue="epsilon", data=agg_results[agg_results.index > 10], palette=pallete)
plt.show()