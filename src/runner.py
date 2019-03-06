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
        self.agent.__init__(k=self.env.n_arms)
        for i in range(1, self.iterations):
            # Gets the action from the agent
            action = self.agent.get_action()

            # Executes action on environment
            reward = self.env.perform_action(action)

            # Gives reward as feedback for agent
            self.agent.set_reward(action, reward)

    def plot_environment(self):
        """
        plots the different distributions
        """
        sns.set()
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(1, 1, 1)

        samples = (self.env.get_samples(10000))
        sns.violinplot(data=samples, ax=ax, palette=self.palette)
        plt.show()

    def plot_selected_actions(self):
        """
        plots the number of times each action was selected
        """
        sns.set()
        fig = plt.figure(figsize=(20, 8))
        ax1 = fig.add_subplot(1, 1, 1)
        sns.barplot(x=list(range(0, self.env.n_arms)), y=self.agent.n_plays, ax=ax1, palette=self.palette)
        plt.show()

    def plot_value_function(self):
        """
        plots the number of times each action was selected
        """
        sns.set()
        fig = plt.figure(figsize=(20, 8))
        ax1 = fig.add_subplot(1, 1, 1)
        samples = self.env.get_samples(10000)
        sns.violinplot(data=samples, ax=ax1, palette=self.palette)
        shift = 0.1
        sns.scatterplot(x=list(np.arange(0 - shift, self.env.n_arms - shift)),
                        y=self.agent.values,
                        ax=ax1,
                        marker='>',
                        s=200, color="black")
        plt.show()
