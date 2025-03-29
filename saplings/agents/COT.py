# Standard library
from typing import List, Optional

# Local
try:
    from saplings.model import Model
    from saplings.abstract import Tool
    from saplings.agents.Base import BaseAgent
    from saplings.dtos import Node, Message
    from saplings.prompts import AGENT_PROMPT
except ImportError:
    from model import Model
    from abstract import Tool, Model
    from agents.Base import BaseAgent
    from dtos import Node, Message
    from prompts import AGENT_PROMPT


class COTAgent(BaseAgent):
    def __init__(
        self,
        tools: List[Tool],
        model: Model,
        prompt: str = AGENT_PROMPT,
        max_depth: int = 5,
        verbose: bool = True,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = False,
        update_prompt: Optional[callable] = None,
    ):
        super().__init__(
            tools,
            model,
            evaluator=None,
            prompt=prompt,
            b_factor=1,
            max_depth=max_depth,
            threshold=1.0,
            verbose=verbose,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            update_prompt=update_prompt,
        )

    def should_terminate(self, node: Node) -> bool:
        return self.is_terminal_node(node)

    async def run_iter_async(self, prompt: str, messages: list[Message] = []):
        self.log(f"Running a ReAct sequence (no search)\n\n\033[37m{prompt}\033[0m\n")

        curr_node = Node([Message.user(prompt)])
        while not self.should_terminate(curr_node):
            async for item in self.expand(curr_node, messages, run_eval=False):
                yield item

            curr_node = curr_node.children[0]

        messages = curr_node.get_trajectory()

        self.log(
            f"\033[1;32mFinal trajectory:\033[0m\n\n"
            + "\n".join(str(m) for m in messages)
        )

        yield messages
