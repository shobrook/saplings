# Standard library
import heapq
from math import inf
from typing import List, Optional, Tuple

# Local
try:
    from saplings.abstract import Tool, Evaluator
    from saplings.agents.Base import BaseAgent
    from saplings.dtos import Message, Node
    from saplings.prompts import AGENT_PROMPT
    from saplings.model import Model
except ImportError:
    from abstract import Tool, Model, Evaluator
    from agents.Base import BaseAgent
    from dtos import Message, Node
    from prompts import AGENT_PROMPT
    from model import Model


class AStarAgent(BaseAgent):
    def __init__(
        self,
        tools: List[Tool],
        model: Model,
        evaluator: Optional[Evaluator] = None,
        prompt: str = AGENT_PROMPT,
        b_factor: int = 3,
        max_depth: int = 5,
        threshold: float = 1.0,
        verbose: bool = True,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = False,
        update_prompt: Optional[callable] = None,
    ):
        super().__init__(
            tools,
            model,
            evaluator,
            prompt,
            b_factor,
            max_depth,
            threshold,
            verbose,
            tool_choice,
            parallel_tool_calls,
            update_prompt,
        )

    def should_terminate(self, node: Node) -> bool:
        return self.is_solution_node(node)

    async def run_iter_async(self, prompt: str, messages: List[Message] = []):
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
            async for item in self.expand(curr_node, messages):
                yield item
            for child in curr_node.children:
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

        yield (messages, score, is_solution)
