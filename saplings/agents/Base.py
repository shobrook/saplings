# Standard library
import asyncio
import threading
from typing import List, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Local
try:
    from saplings.model import Model
    from saplings.dtos import Message, Node
    from saplings.evaluator import Evaluator
    from saplings.prompts import AGENT_PROMPT
    from saplings.abstract import Tool, Evaluator as BaseEvaluator
except ImportError:
    from model import Model
    from dtos import Message, Node
    from evaluator import Evaluator
    from prompts import AGENT_PROMPT
    from abstract import Tool, Evaluator as BaseEvaluator


class BaseAgent(object):
    def __init__(
        self,
        tools: List[Tool],
        model: Model,
        evaluator: Optional[BaseEvaluator] = None,
        prompt: str = AGENT_PROMPT,
        b_factor: int = 3,
        max_depth: int = 5,
        threshold: float = 1.0,
        verbose: bool = True,
        tool_choice: str = "auto",  # or "required"
        parallel_tool_calls: bool = False,
        update_prompt: Optional[callable] = None,
    ):
        self.tools = tools
        self.model = model
        self.evaluator = evaluator if evaluator else Evaluator(model)
        self.prompt = prompt  # Governs tool calls
        self.b_factor = b_factor  # Branching factor
        self.max_depth = max_depth
        self.threshold = threshold  # Solution threshold
        self.verbose = verbose  # For debugging
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.max_tool_call_tokens = 2048
        self.update_system_prompt = update_prompt if update_prompt else lambda t: prompt

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

        # TODO: Should we only consider active tools?
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

    def get_tool_by_name(self, name: str) -> Tool:
        """
        Gets a tool object by its name.
        """

        for tool in self.tools:
            if tool.name == name:
                return tool

        raise ValueError(f"Tool with name '{name}' not found.")

    def update_prompts(self, trajectory: List[Message]):
        """
        Updates tool prompts and system prompt based on the current trajectory.
        """

        self.prompt = self.update_system_prompt(trajectory)
        for tool in self.tools:
            tool.update_definition(trajectory)

    async def generate_candidates(
        self, node: Node, messages: List[Message], n: Optional[int] = None
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

        # Get active tools
        trajectory = messages + node.get_trajectory()
        tools = [tool for tool in self.tools if tool.is_active(trajectory)]
        tool_schemas = [tool.get_schema() for tool in tools]

        # Generate tool calls
        system_message = Message.system(self.prompt)
        headroom = (
            self.model.count_message_tokens(system_message) + self.max_tool_call_tokens
        )
        messages = [system_message] + self.model.truncate_messages(
            trajectory, headroom, tools
        )
        response = await self.model.run_async(
            messages,
            tools=tool_schemas,
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
            return None  # TODO: When would this be hit?

        fn_call = message.tool_calls[0]
        tool = self.get_tool_by_name(fn_call.name)
        output = await tool.run(**fn_call.arguments, trajectory=trajectory)
        formatted_output = tool.format_output(output)
        tool_response = Message.tool(formatted_output, fn_call.id, raw_output=output)
        tool_response.parent_id = message.parent_id
        tool_response.id = message.id

        return tool_response

    async def evaluate(self, node: Node, messages: List[Message] = []) -> Node:
        """
        Evaluates a node in the search tree. If a custom evaluator is not provided,
        the LLM self-evaluates the node.
        """

        # TODO: If self.is_output_node(node), should we only evaluate the message(s) in that node?
        # Or evaluate the whole trajectory? For now, we are evaluating the whole trajectory.

        trajectory = messages + node.get_trajectory()
        evaluation = await self.evaluator.run(trajectory)
        node.set_evaluation(evaluation)

        # TODO: Add a self-consistency term. We can do this by sampling multiple outputs for
        # the candidate generation (the step before evaluation) and weighing each candidate
        # by the probability of it being generated. Then the final value would be
        # V(n) = LLM(n) * lambda + SC(n) * (1 - lambda), where lambda is a hyperparameter
        # that controls the weight of the self-consistency term.

        return node

    async def expand(self, node: Node, messages: List[Message], run_eval=True):
        if self.is_terminal_node(node):
            self.log(f"\033[1;31mReached terminal node\033[0m\n\n{node}\n")
            yield []
            return

        self.log(f"Expanding node\n\n{node}\n")

        # Update prompts based on current trajectory
        trajectory = messages + node.get_trajectory()
        self.update_prompts(trajectory)

        # Create (partial) child nodes
        children = []

        # Generate candidate next tool calls, execute each
        tool_calls = await self.generate_candidates(node, messages)
        tasks = []
        for tool_call in tool_calls:
            partial_child = Node([tool_call], parent=node)
            children.append(partial_child)

            tool_call.id = partial_child.id
            tool_call.parent_id = (
                partial_child.parent.id if partial_child.parent else None
            )
            yield tool_call

            task = self.execute_tool_call(tool_call, trajectory)
            tasks.append(task)

        id_to_index = {child.id: i for i, child in enumerate(children)}

        tool_responses = [None] * len(tasks)
        for task in asyncio.as_completed(tasks):
            tool_response = await task
            if not tool_response:
                continue

            j = id_to_index[tool_response.id]
            child = children[j]
            child.messages.append(tool_response)
            tool_responses[j] = tool_response

            if not run_eval:
                yield tool_response

        # Evaluate each child
        if run_eval:
            tasks = [self.evaluate(child, messages) for child in children]
            for result in asyncio.as_completed(tasks):
                child = await result
                tool_response = child.messages[-1]
                tool_response.score = child.score
                yield tool_response

        self.log(
            f"Generated {len(children)} children\n\n"
            + "\n\n".join(str(child) for child in children)
            + "\n"
        )

        # Grow the tree
        node.add_children(children)

    async def run_async(self, prompt: str, messages: list[Message] = []):
        last_item = None
        async for item in self.run_iter_async(prompt, messages):
            last_item = item

        return last_item

    def run(self, prompt: str, messages: List[Message] = [], **kwargs) -> any:
        loop = asyncio.new_event_loop()
        result = None

        def _run():
            nonlocal result
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.run_async(prompt, messages, **kwargs))
            loop.close()

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()
        return result

    def run_iter(self, prompt: str, messages: List[Message] = [], **kwargs) -> any:
        async def run_async_wrapper():
            async for item in self.run_iter_async(prompt, messages):
                yield item

        # Get an event loop and run the async generator
        loop = asyncio.get_event_loop()
        async_gen = run_async_wrapper()

        try:
            while True:
                # Get the next item from the async generator
                try:
                    item = loop.run_until_complete(async_gen.__anext__())
                    yield item
                except StopAsyncIteration:
                    break
        finally:
            # Clean up the async generator
            loop.run_until_complete(async_gen.aclose())

    async def call_tool_async(
        self, tool_name: str, messages: List[Message] = []
    ) -> Message:
        system_message = Message.system(self.prompt)
        headroom = (
            self.model.count_message_tokens(system_message) + self.max_tool_call_tokens
        )
        messages = [system_message] + self.model.truncate_messages(
            messages, headroom, self.tools
        )
        response = await self.model.run_async(
            messages,
            tools=[tool.get_schema() for tool in self.tools],
            parallel_tool_calls=self.parallel_tool_calls,
            tool_choice={"type": "function", "function": {"name": tool_name}},
            max_tokens=self.max_tool_call_tokens,
            n=1,
            temperature=1.0,
        )
        return Message.from_openai_message(response)

    async def run_tool_async(
        self, tool_call: Message, messages: List[Message] = []
    ) -> Message:
        return await self.execute_tool_call(tool_call, messages)

    def call_tool(self, tool_name: str, messages: List[Message] = []) -> Message:
        loop = asyncio.new_event_loop()
        result = None

        def _run():
            nonlocal result
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.call_tool_async(tool_name, messages)
                )
            finally:
                # Clean up pending tasks before closing
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()
        return result

    def run_tool(self, tool_call: Message, messages: List[Message] = []) -> Message:
        loop = asyncio.new_event_loop()
        result = None

        def _run():
            nonlocal result
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.run_tool_async(tool_call, messages)
                )
            finally:
                # Clean up pending tasks before closing
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()
        return result
