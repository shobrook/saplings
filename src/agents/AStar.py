# Standard library
import heapq
from math import inf
from typing import List, Optional, Tuple

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
        threshold: float = 1.0,
        verbose: bool = True,
    ):
        super().__init__(
            tools, model, evaluator, prompt, b_factor, max_depth, threshold, verbose
        )

    def should_terminate(self, node: Node) -> bool:
        return self.is_solution_node(node)

    async def run_async(self, prompt: str) -> Tuple[List[Message], float, bool]:
        # Max priority queue
        root_node = Node([Message.user(prompt)])
        best_score = -inf  # Negative scores for max behavior
        frontier = []

        # Push the initial node to the frontier
        heapq.heappush(frontier, (0, root_node))

        self.log(f"Running an A* search\n\n\033[37m{prompt}\033[0m\n")
        while frontier:
            # Get the next node to explore
            neg_score, curr_node = heapq.heappop(frontier)
            curr_score = -neg_score  # Convert back to positive score

            # Update the best score
            if curr_score > best_score:
                best_score = curr_score

            # Stop search if current node is a solution
            if self.should_terminate(curr_node):
                self.log("\033[1;32mFound a solution! Terminating search.\033[0m")
                break

            # Expand the current node, add children to the frontier
            children = await self.expand(curr_node)
            for child in children:
                heapq.heappush(frontier, (-child.score, child))
        else:
            self.log(
                "\033[1;31mNo solution found. Returning the best trajectory available.\033[0m"
            )

        best_node = self.get_best_node(root_node)
        messages, score, is_solution = (
            best_node.get_trajectory(),
            best_node.score,
            self.is_solution_node(best_node),
        )

        self.log(
            f"\033[1;32mBest trajectory (score={score}, is_solution={is_solution}):\033[0m\n\n"
            + "\n".join(str(m) for m in messages)
        )

        return messages, score, is_solution
