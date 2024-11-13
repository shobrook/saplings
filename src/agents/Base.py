# Standard library
import asyncio
from collections import defaultdict
from typing import List, Optional

# Local
from src.abstract import Model, Tool
from src.dtos import Evaluation, Message, Node
from src.llms import OpenAI
from src.prompts import AGENT_PROMPT, EVAL_PROMPT


class BaseAgent(object):
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
        tool_choice: str = "auto",  # or "required"
    ):
        # Setup
        self.tools = tools
        self.model = model if model else OpenAI()
        self.evaluator = evaluator
        self.prompt = prompt  # Governs tool calls
        self.b_factor = b_factor  # Branching factor
        self.max_depth = max_depth
        self.threshold = threshold  # Solution threshold
        self.verbose = verbose  # For debugging
        self.tool_choice = tool_choice

        # Token limits (allow us to do automatic token budgeting)
        self.tool_call_headroom = 2048
        self.eval_headroom = 1024

    def log(self, message: str):
        if not self.verbose:
            return

        print("LOG: " + message)

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

        if self.tool_choice == "optional":
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

    def get_trimmed_trajectory(self, node: Node, headroom: int) -> List[Message]:
        """
        Gets the search branch the node belongs to. Trims + drops messages to
        fit within the token headroom.
        """

        # Get *all* messages in the search branch
        messages = node.get_trajectory(include_evals=False)
        input_message = messages[0]

        # Trim messages to fit within the token headroom
        headroom = self.model.get_context_window() - headroom
        token_count = self.model.count_message_tokens(input_message)
        token_count += sum(self.model.count_tool_tokens(tool) for tool in self.tools)

        trimmed_messages = [input_message]
        for message in reversed(messages[1:]):
            num_tokens = self.model.count_message_tokens(message)
            if token_count + num_tokens > headroom:
                if message.role == "tool":
                    message.content = "[HIDDEN]"
                    num_tokens = self.model.count_message_tokens(message)

                    if token_count + num_tokens <= headroom:
                        token_count += num_tokens
                        trimmed_messages.insert(1, message)
                        continue

                break

            token_count += num_tokens
            trimmed_messages.insert(1, message)

        return trimmed_messages

    async def generate_candidates(
        self, node: Node, n: Optional[int] = None
    ) -> List[Message]:
        """
        Generates plausible next actions to take in a given trajectory.
        Obtains `b_factor` candidate actions by using the `num_outputs`
        parameter to control the number of tool calls made by the LLM.

        If `num_outputs` > `b_factor`, we sort the tool calls by frequency
        (as in # of duplicates) and return the top `b_factor` tool calls.

        Candidates are always unique/de-duplicated.
        """

        # No. of candidates to generate
        n = n if n else self.b_factor

        # Generate tool calls
        system_message = Message.system(self.prompt)
        headroom = (
            self.model.count_message_tokens(system_message) + self.tool_call_headroom
        )
        messages = [system_message] + self.get_trimmed_trajectory(node, headroom)
        response = await self.model.arun(
            messages,
            tools=self.get_tool_schemas(),
            parallel_tool_calls=False,
            tool_choice=self.tool_choice,
            max_tokens=self.tool_call_headroom,
            n=n,
            temperature=1.0,
        )
        if n == 1:
            candidates = [Message.from_response(response)]
        else:
            candidates = [Message.from_response(choice.message) for choice in response]

        # TODO: Handle non-tool call candidates

        # Deduplicate tool calls and sort by frequency
        tool_counts = defaultdict(lambda: 0)
        tool_messages = defaultdict(dict)
        response_messages = []
        for message in candidates:
            if not message.tool_calls:
                response_messages.append(message)
                continue

            tool_call = message.tool_calls[0]
            tool_counts[hash(tool_call)] += 1
            tool_messages[hash(tool_call)] = message

        # TODO: Sort by tool name as tiebreaker

        top_tools = sorted(tool_counts.items(), key=lambda item: item[1], reverse=True)
        top_tools = [tool_messages[tool] for tool, _ in top_tools]
        candidates = response_messages + top_tools

        return candidates[: self.b_factor]

    async def execute_tool_call(self, message: Message) -> Message:
        if not message.tool_calls:
            return None

        fn_call = message.tool_calls[0]
        tool = self.get_tool_by_name(fn_call.name)
        output = await tool.run(**fn_call.arguments)
        formatted_output = tool.format_output(output)
        tool_response = Message.tool(formatted_output, fn_call.id)

        # TODO: Store raw output somewhere

        return tool_response

    async def evaluate(self, node: Node) -> Node:
        """
        Evaluates a node in the search tree. If a custom evaluator is not provided,
        the LLM self-evaluates the node.
        """

        # Use user-provided evaluator
        if self.evaluator:
            evaluation = self.evaluator(node)
            node.set_evaluation(evaluation)
            return node

        # TODO: If self.is_output_node(node), should we only evaluate the message(s) in that node?
        # Or evaluate the whole trajectory? For now, we are evaluating the whole trajectory.

        # Use default evaluator
        system_message = Message.system(EVAL_PROMPT)
        headroom = self.model.count_message_tokens(system_message) + self.eval_headroom
        messages = [system_message] + self.get_trimmed_trajectory(node, headroom)
        response = await self.model.arun(
            messages,
            max_tokens=self.eval_headroom,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "Your thoughts and reasoning process. Keep it brief and concise.",
                            },
                            "score": {
                                "type": "number",
                                "description": "Score from 0-10 on the quality of the trajectory. A 10 indicates that the agent has completely succeeded in satisfying the user's intent.",
                            },
                        },
                        "required": ["reasoning", "score"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        response = Message.from_response(response)
        evaluation = Evaluation.from_message(response)
        node.set_evaluation(evaluation)

        return node

    async def expand(self, node: Node) -> List[Node]:
        if self.is_terminal_node(node):
            self.log(f"Skipping expansion of terminal node:\n{node}\n")
            return []

        self.log(f"Expanding node:\n{node}\n")

        # Generate candidate next tool calls, execute each
        # num_candidates = max(self.b_factor * len(self.tools), 20)
        tool_calls = await self.generate_candidates(node)  # , num_candidates)
        tasks = [self.execute_tool_call(tool_call) for tool_call in tool_calls]
        tool_responses = await asyncio.gather(*tasks)

        # Create child nodes
        children = [
            Node([call, response] if response else [call], parent=node)
            for call, response in zip(tool_calls, tool_responses)
        ]

        # Evaluate each child
        tasks = [self.evaluate(child) for child in children]
        await asyncio.gather(*tasks)

        self.log(
            f"Generated {len(children)} children nodes:"
            + "\n"
            + "\n".join(str(child) for child in children)
            + "\n"
        )

        # TODO: Add a self-consistency term

        # Grow the tree
        node.add_children(children)

        return children
