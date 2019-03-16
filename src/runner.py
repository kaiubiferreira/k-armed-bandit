import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Runner:
    def __init__(self, env, agent, iterations=1000):
        """
        Initializes an instance to run the simulation and collect some statistics
        :param env: KArmedBandit Environment
        :param agent: KArmedBandit Agent
        :param iterations: Number of times the agent will select actions
        """
        self.env = env
        self.agent = agent
        self.iterations = iterations
        self.palette = sns.color_palette("muted", n_colors=self.env.n_arms)

    def run(self):
        """
        Runs the simulation
        """
        # Resets agent status
        self.agent.__init__(k=self.env.n_arms, n_bandits=self.env.n_bandits)
        for i in range(1, self.iterations):
            # Gets the action from the agent
            current_state = self.env.latest_state
            action = self.agent.get_action(current_state)

            # Executes action on environment
            reward, next_state = self.env.perform_action(action)

            # Gives reward as feedback for agent
            self.agent.set_reward(current_state, action, reward)

    def plot_environment(self):
        """
        plots the different distributions
        """
        sns.set()
        fig = plt.figure(figsize=(20, 8))

        for bandit in range(0, self.env.n_bandits):
            ax = fig.add_subplot(self.env.n_bandits, 1, bandit + 1)

            samples = (self.env.bandits[bandit].get_samples(10000))
            sns.violinplot(data=samples, ax=ax, palette=self.palette)
        plt.show()

    def plot_selected_actions(self):
        """
        plots the number of times each action was selected
        """
        sns.set()
        fig = plt.figure(figsize=(20, 8))

        for bandit in range(0, self.env.n_bandits):
            ax = fig.add_subplot(self.env.n_bandits, 1, bandit + 1)
            sns.barplot(x=list(range(0, self.env.n_arms)), y=self.agent.n_plays[bandit], ax=ax, palette=self.palette)

        plt.show()

    def plot_value_function(self):
        """
        plots the number of times each action was selected
        """
        sns.set()
        fig = plt.figure(figsize=(20, 8))

        for bandit in range(0, self.env.n_bandits):
            ax = fig.add_subplot(self.env.n_bandits, 1, bandit + 1)
            samples = self.env.bandits[bandit].get_samples(10000)
            sns.violinplot(data=samples, ax=ax, palette=self.palette)
            shift = 0.1
            sns.scatterplot(x=list(np.arange(0 - shift, self.env.n_arms - shift)),
                            y=self.agent.values[bandit],
                            ax=ax,
                            marker='>',
                            s=200, color="black")
        plt.show()
