from src.agent import KArmedBanditAgent
from src.environment import KArmedBanditEnvironment
from src.runner import Runner

env = KArmedBanditEnvironment(k=10)
agent = KArmedBanditAgent(k=env.n_arms)
runner = Runner(env, agent)
runner.run()
runner.plot_environment()
runner.plot_selected_actions()
runner.plot_value_function()
