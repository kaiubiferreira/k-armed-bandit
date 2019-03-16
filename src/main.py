from src.agent import KArmedBanditAgent
from src.environment import KArmedBanditEnvironment
from src.runner import Runner

k = 10
n_bandits = 3

env = KArmedBanditEnvironment(k=k, n_bandits=n_bandits)
agent = KArmedBanditAgent(k=k, n_bandits=n_bandits)
runner = Runner(env, agent, iterations=5000)
runner.run()
runner.plot_environment()
runner.plot_selected_actions()
runner.plot_value_function()
