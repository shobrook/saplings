# Standard library
import asyncio
import threading
from collections import defaultdict
from typing import List, Optional

# Local
try:
    from saplings.evaluator import Evaluator
    from saplings.abstract import Model, Tool, Evaluator as BaseEvaluator
    from saplings.dtos import Message, Node
    from saplings.llms import OpenAI
    from saplings.prompts import AGENT_PROMPT
except ImportError:
    from evaluator import Evaluator
    from abstract import Model, Tool, Evaluator as BaseEvaluator
    from dtos import Message, Node
    from llms import OpenAI
    from prompts import AGENT_PROMPT


class BaseAgent(object):
    def __init__(
        self,
        tools: List[Tool],
        model: Optional[Model] = None,
        evaluator: Optional[BaseEvaluator] = None,
        prompt: str = AGENT_PROMPT,
        b_factor: int = 3,
        max_depth: int = 5,
        threshold: float = 1.0,
        verbose: bool = True,
        tool_choice: str = "auto",  # or "required"
        parallel_tool_calls: bool = False,
    ):
        self.tools = tools
        self.model = model if model else OpenAI()
        self.evaluator = evaluator if evaluator else Evaluator(model)
        self.prompt = prompt  # Governs tool calls
        self.b_factor = b_factor  # Branching factor
        self.max_depth = max_depth
        self.threshold = threshold  # Solution threshold
        self.verbose = verbose  # For debugging
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.max_tool_call_tokens = 2048

    def log(self, message: str):
        if not self.verbose:
            return

        bold_yellow = "\033[1;33m"
        reset = "\033[0m"

        print(f"{bold_yellow}SAPLINGS LOG:{reset} {message}")

    def is_output_node(self, node: Node) -> bool:
        """
        Checks if a node represents a final response to the user's prompt.
        """

        for message in node.messages:
            if message.role != "assistant":
                continue

            if not message.tool_calls:  # and len(node.messages) == 1
                return True

            for tool_call in message.tool_calls:
                tool = self.get_tool_by_name(tool_call.name)
                if tool.is_terminal:
                    return True

        return False

    def is_terminal_node(self, node: Node) -> bool:
        if self.is_solution_node(node):
            return True

        if self.is_output_node(node):
            return True

        if node.depth >= self.max_depth:
            return True

        return False

    def is_solution_node(self, node: Node) -> bool:
        # NOTE: If the agent *can* generate an output, then even if a
        # score is above the threshold, we do not consider it a solution
        # unless the node is an output node.

        if node.score >= self.threshold:
            if not self.can_generate_output():
                return True

            if self.is_output_node(node):
                return True

        return False

    def can_generate_output(self) -> bool:
        """
        Checks if the agent can generate a direct output for the user. This means it
        either has a tool marked as `is_output` or it has a tool choice of `optional`,
        which means the model can choose to generate a response.
        """

        for tool in self.tools:
            if tool.is_terminal:
                return True

        if self.tool_choice == "auto":
            return True

        return False

    def get_best_node(self, root: Node) -> Node:
        """
        Gets the best solution from the search tree.

        If a search terminated before a solution node was found,
        this will return the node with the highest score.

        If an agent can generate an output, we'll prioritize output nodes.
        If not, we consider all leaf nodes.
        """

        best_score, best_output_score = 0, 0
        best_node, best_output_node = root, None
        for node in root.bfs():
            if not node.is_leaf:
                continue

            if self.is_output_node(node):
                if node.score >= best_output_score:
                    best_output_score, best_output_node = node.score, node

            if node.score >= best_score:
                best_score, best_node = node.score, node

        if best_output_node:
            return best_output_node

        return best_node

    def get_tool_schemas(self) -> List[dict]:
        """
        Used to prepare tools for the LLM.
        """

        return [tool.get_schema() for tool in self.tools]

    def get_tool_by_name(self, name: str) -> Tool:
        """
        Gets a tool object by its name.
        """

        for tool in self.tools:
            if tool.name == name:
                return tool

        raise ValueError(f"Tool with name '{name}' not found.")

    async def generate_candidates(
        self, node: Node, n: Optional[int] = None
    ) -> List[Message]:
        """
        Generates plausible next tool calls to take in a given trajectory.
        Obtains `b_factor` candidate tool calls by using the `num_outputs`
        parameter to control the number of tool calls made by the LLM.
        Tool calls are always unique/de-duplicated. Can also be a response
        not a tool call if `tool_choice == "auto"`.
        """

        # No. of candidates to generate
        n = n if n else self.b_factor

        # Generate tool calls
        system_message = Message.system(self.prompt)
        headroom = (
            self.model.count_message_tokens(system_message) + self.max_tool_call_tokens
        )
        messages = [system_message] + self.model.truncate_messages(
            node.get_trajectory(), headroom, self.tools
        )
        response = await self.model.run_async(
            messages,
            tools=self.get_tool_schemas(),
            parallel_tool_calls=self.parallel_tool_calls,
            tool_choice=self.tool_choice,
            max_tokens=self.max_tool_call_tokens,
            n=n,
            temperature=1.0,
        )
        if n == 1:
            candidates = [Message.from_openai_message(response)]
        else:
            candidates = [
                Message.from_openai_message(choice.message) for choice in response
            ]

        # Deduplicate tool calls and sort by frequency
        tool_counts = defaultdict(lambda: 0)
        tool_messages = defaultdict(dict)
        response_messages = set()
        for message in candidates:
            if not message.tool_calls:
                response_messages.add(message)
                continue

            tool_call = message.tool_calls[0]
            tool_counts[hash(tool_call)] += 1
            tool_messages[hash(tool_call)] = message

        # TODO: Sort by tool name as tiebreaker

        top_tools = sorted(tool_counts.items(), key=lambda item: item[1], reverse=True)
        top_tools = [tool_messages[tool] for tool, _ in top_tools]
        candidates = list(response_messages) + top_tools  # We prioritize responses

        return candidates[: self.b_factor]

    async def execute_tool_call(
        self, message: Message, trajectory: List[Message]
    ) -> Message:
        if not message.tool_calls:
            return None

        fn_call = message.tool_calls[0]
        tool = self.get_tool_by_name(fn_call.name)
        output = await tool.run(**fn_call.arguments, trajectory=trajectory)
        formatted_output = tool.format_output(output)
        tool_response = Message.tool(formatted_output, fn_call.id, raw_output=output)

        return tool_response

    async def evaluate(self, node: Node) -> Node:
        """
        Evaluates a node in the search tree. If a custom evaluator is not provided,
        the LLM self-evaluates the node.
        """

        # TODO: If self.is_output_node(node), should we only evaluate the message(s) in that node?
        # Or evaluate the whole trajectory? For now, we are evaluating the whole trajectory.

        trajectory = node.get_trajectory()
        evaluation = await self.evaluator.run(trajectory)
        node.set_evaluation(evaluation)

        # TODO: Add a self-consistency term. We can do this by sampling multiple outputs for
        # the candidate generation (the step before evaluation) and weighing each candidate
        # by the probability of it being generated. Then the final value would be
        # V(n) = LLM(n) * lambda + SC(n) * (1 - lambda), where lambda is a hyperparameter
        # that controls the weight of the self-consistency term.

        return node

    async def expand(self, node: Node, run_eval=True) -> List[Node]:
        if self.is_terminal_node(node):
            self.log(f"\033[1;31mReached terminal node\033[0m\n\n{node}\n")
            return []

        self.log(f"Expanding node\n\n{node}\n")

        # Generate candidate next tool calls, execute each
        trajectory = node.get_trajectory()
        tool_calls = await self.generate_candidates(node)
        tasks = [
            self.execute_tool_call(tool_call, trajectory) for tool_call in tool_calls
        ]
        tool_responses = await asyncio.gather(*tasks)

        # Create child nodes
        children = [
            Node([call, response] if response else [call], parent=node)
            for call, response in zip(tool_calls, tool_responses)
        ]

        # Evaluate each child
        if run_eval:
            tasks = [self.evaluate(child) for child in children]
            await asyncio.gather(*tasks)

        self.log(
            f"Generated {len(children)} children\n\n"
            + "\n\n".join(str(child) for child in children)
            + "\n"
        )

        # Grow the tree
        node.add_children(children)

        return children

    def run(self, prompt: str, **kwargs) -> any:
        loop = asyncio.new_event_loop()
        result = None

        def _run():
            nonlocal result
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.run_async(prompt, **kwargs))
            loop.close()

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()
        return result
