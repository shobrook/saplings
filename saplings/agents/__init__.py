try:
    from saplings.agents.AStar import AStarAgent
    from saplings.agents.Greedy import GreedyAgent
    from saplings.agents.MonteCarlo import MonteCarloAgent
    from saplings.agents.COT import COTAgent
except ImportError:
    from .AStar import AStarAgent
    from .Greedy import GreedyAgent
    from .MonteCarlo import MonteCarloAgent
    from .COT import COTAgent
