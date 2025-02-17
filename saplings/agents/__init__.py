try:
    from saplings.agents.AStar import AStarAgent
    from saplings.agents.Greedy import GreedyAgent
    from saplings.agents.MonteCarlo import MonteCarloAgent
    from saplings.agents.COT import COTAgent
except ImportError:
    from agents.AStar import AStarAgent
    from agents.Greedy import GreedyAgent
    from agents.MonteCarlo import MonteCarloAgent
    from agents.COT import COTAgent
