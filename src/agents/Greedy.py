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
        curr_node = Node([Message.user(prompt)])

        while not self.should_terminate(curr_node):
            await self.expand(curr_node)
            curr_node = self.get_best_node(curr_node)

        return curr_node
