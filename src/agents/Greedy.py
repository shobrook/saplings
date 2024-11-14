# Standard library
from typing import List, Optional

# Local
from src.abstract import Tool, Model
from src.agents.Base import BaseAgent
from src.dtos import Node, Message
from src.prompts import AGENT_PROMPT


class GreedyAgent(BaseAgent):
    def __init__(
        self,
        tools: List[Tool],
        model: Optional[Model] = None,
        evaluator: Optional[any] = None,
        prompt: str = AGENT_PROMPT,
        b_factor: int = 3,
        max_depth: int = 5,
        threshold: float = 1.0,
    ):
        super().__init__(
            tools, model, evaluator, prompt, b_factor, max_depth, threshold
        )

    def should_terminate(self, node: Node) -> bool:
        return self.is_terminal_node(node)

    async def run(self, prompt: str) -> Node:
        self.log(f"Running a greedy best-first search\n\n\033[37m{prompt}\033[0m\n")

        best_node = Node([Message.user(prompt)])
        while not self.should_terminate(best_node):
            await self.expand(best_node)
            best_node = self.get_best_node(best_node)

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
