try:
    from saplings.model import Model
    from saplings.evaluator import Evaluator
    from saplings.agents import AStarAgent, GreedyAgent, MonteCarloAgent, COTAgent
except ImportError:
    from model import Model
    from evaluator import Evaluator
    from agents import AStarAgent, GreedyAgent, MonteCarloAgent, COTAgent
