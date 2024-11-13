# Standard library
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
        tool_choice: str = "required",  # or "optional"
        b_factor: int = 3,
    ):
        # Setup
        self.tools = tools
        self.model = model if model else OpenAI()
        self.evaluator = evaluator
        self.prompt = prompt
        self.tool_choice = tool_choice
        self.b_factor = b_factor  # Branching factor

        # Token limits (TODO: Should these even exist?)
        self.tool_call_headroom = 1024
        self.eval_headroom = 512

    def agent_has_output_tools(self) -> bool:
        """
        Checks if the agent has any tools that can generate a response, or if tool use
        is optional and the model can generate a response. If so, then it's possible
        for search nodes to be terminal without being solutions.
        """

        for tool in self.tools:
            if tool.is_terminal:
                return True

        if self.tool_choice == "optional":
            return True

        return False

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

    def get_best_node(self, root: Node) -> Node:
        """
        Gets the best solution from the search tree.

        1. If agent has output tools, then we only consider output nodes (e.g. responses).
        2. If agent has no output tools, then we consider all leaf nodes.
        """

        best_score, best_node = 0, root
        for node in root.bfs():
            if self.agent_has_output_tools():
                if not self.is_output_node(node):
                    continue
            elif not node.is_leaf:
                continue

            if node.normalized_score > best_score:
                best_score, best_node = node.normalized_score, node

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

    def get_trajectory(self, node: Node, headroom: int) -> List[Message]:
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
        for message in reversed(messages):
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

        # TODO: If node.is_leaf(), should we only evaluate the messages in that node?
        # Or evaluate the whole trajectory?

        # Use default evaluator
        system_message = Message.system(EVAL_PROMPT)
        headroom = self.model.count_message_tokens(system_message) + self.eval_headroom
        messages = [system_message] + self.get_trajectory(node, headroom)
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
                            # "is_solved": {
                            #     "type": "boolean",
                            #     "description": "Whether the agent has succeeded in satisfying the user's intent.",
                            # },
                        },
                    },
                },
            },
        )
        response = Message.from_response(response)
        evaluation = Evaluation.from_message(response)
        node.set_evaluation(evaluation)

        return node

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

        # Generate tool calls
        system_message = Message.system(self.prompt)
        headroom = (
            self.model.count_message_tokens(system_message) + self.tool_call_headroom
        )
        messages = [system_message] + self.get_trajectory(node, headroom)
        response = await self.model.arun(
            messages,
            tools=self.get_tool_schemas(),
            parallel_tool_calls=False,
            tool_choice=self.tool_choice,
            max_tokens=self.tool_call_headroom,
            n=n if n else self.b_factor,
            temperature=1.0,
        )
        candidates = [Message.from_response(choice) for choice in response]

        # TODO: Handle non-tool call candidates

        # Deduplicate tool calls and sort by frequency
        tool_counts = defaultdict(lambda: 0)
        tool_messages = defaultdict(dict)
        for message in candidates:
            tool_call = message.tool_calls[0]
            tool_counts[hash(tool_call)] += 1
            tool_messages[hash(tool_call)] = message

        # TODO: Sort by tool name as tiebreaker

        top_tools = sorted(tool_counts.items(), key=lambda item: item[1], reverse=True)
        top_tools = [tool_messages[tool] for tool, _ in top_tools]

        return top_tools[: self.b_factor]

    async def execute_tool_call(self, tool_call: Message) -> Message:
        fn_call = tool_call.tool_calls[0]
        tool = self.get_tool_by_name(fn_call.name)
        output = await tool.run(**fn_call.arguments)
        formatted_output = tool.format_output(output)
        tool_response = Message.tool(formatted_output, fn_call.id)

        # TODO: Store raw output somewhere

        return tool_response
