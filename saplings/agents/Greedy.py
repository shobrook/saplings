# Standard library
from typing import List, Optional, Tuple

# Local
try:
    from saplings.dtos import Node, Message
    from saplings.prompts import AGENT_PROMPT
    from saplings.agents.Base import BaseAgent
    from saplings.abstract import Tool, Model, Evaluator
except ImportError:
    from dtos import Node, Message
    from prompts import AGENT_PROMPT
    from agents.Base import BaseAgent
    from abstract import Tool, Model, Evaluator


class GreedyAgent(BaseAgent):
    def __init__(
        self,
        tools: List[Tool],
        model: Optional[Model] = None,
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
        return self.is_terminal_node(node)

    async def run_async(
        self, prompt: str, messages: List[Message] = []
    ) -> Tuple[List[Message], float, bool]:
        self.log(f"Running a greedy best-first search\n\n\033[37m{prompt}\033[0m\n")

        best_node = Node([Message.user(prompt)])
        while not self.should_terminate(best_node):
            await self.expand(best_node, messages)
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
