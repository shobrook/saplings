try:
    from saplings.agents import AStarAgent, GreedyAgent, MonteCarloAgent, COTAgent
    from saplings.evaluator import Evaluator
except ImportError:
    from agents import AStarAgent, GreedyAgent, MonteCarloAgent, COTAgent
    from evaluator import Evaluator
