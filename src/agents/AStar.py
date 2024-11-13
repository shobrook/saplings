# Standard library
import heapq
from math import inf
from typing import List, Optional

# Local
from src.abstract import Tool, Model
from src.agents.Base import BaseAgent
from src.dtos import Message, Node
from src.prompts import AGENT_PROMPT


class AStarAgent(BaseAgent):
    def __init__(
        self,
        tools: List[Tool],
        model: Optional[Model] = None,
        evaluator: Optional[any] = None,
        prompt: str = AGENT_PROMPT,
        b_factor: int = 3,
        max_depth: int = 5,
        threshold: float = 0.7,
    ):
        super().__init__(
            tools, model, evaluator, prompt, b_factor, max_depth, threshold
        )

    def should_terminate(self, node: Node) -> bool:
        return self.is_solution_node(node)

    async def run(self, prompt: str) -> Node:
        # Max priority queue
        root_node = Node([Message.user(prompt)])
        best_score = -inf  # Negative scores for max behavior
        frontier = []

        # Push the initial node to the frontier
        heapq.heappush(frontier, (0, root_node))

        # Begin the search
        while frontier:
            # Get the next node to explore
            neg_score, curr_node = heapq.heappop(frontier)
            curr_score = -neg_score  # Convert back to positive score

            # Update the best score
            if curr_score > best_score:
                best_score = curr_score

            # Stop search if current node is a solution
            if self.should_terminate(curr_node):
                break

            # Expand the current node, add children to the frontier
            children = await self.expand(curr_node)
            for child in children:
                heapq.heappush(frontier, (-child.score, child))

        return self.get_best_node(root_node)
