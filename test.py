# Local
try:
    from saplings import AStarAgent, GreedyAgent, MonteCarloAgent, COTAgent, Model
    from saplings.examples import (
        AdditionTool,
        SubtractionTool,
        MultiplicationTool,
        DivisionTool,
    )
except ImportError:
    from agents import AStarAgent, GreedyAgent, MonteCarloAgent, COTAgent
    from model import Model
    from examples import (
        AdditionTool,
        SubtractionTool,
        MultiplicationTool,
        DivisionTool,
    )

agent = AStarAgent(
    model=Model("openai/gpt-4o"),
    tools=[
        # ExaSearchTool(),
        MultiplicationTool(),
        SubtractionTool(),
    ],
    max_depth=4,
    verbose=False,
)
for message in agent.run_iter(
    "Let x = 9418.343 * 8.11 and y = 2x. Calculate (xy)(x^2)."
):
    print("MESSAGE:")
    print(message)
    print()
