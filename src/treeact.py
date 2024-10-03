# Standard library
import heapq
from math import inf
from typing import List, Union

# Third party
import json_repair

# Local
from src.llms import OpenAI
from src.abstract import Tool
from src.dtos import State, Message
from src.prompts import EVAL_PROMPT, AGENT_PROMPT

# Hyperparameters:
# - depth
# - branching factor
# - search budget
# - termination threshold


#########
# HELPERS
#########


def assert_valid_tools(tools: List[Tool]):
    names = set()
    for tool in tools:
        assert tool.name not in names, f"Duplicate tool name: {tool.name}"
        names.add(tool.name)

        assert isinstance(
            tool, Tool
        ), f"Tool must be an instance of treeact.abstract.Tool"


def assert_valid_threshold(threshold: float):
    assert 0 <= threshold <= 1, "Threshold must be between 0 and 1"


######
# MAIN
######


class TreeActAgent(object):
    """
    depth : int
        Depth
    b_factor : int
        Branching factor
    budget : int
        Search budget, which determines the maximum size of the search tree
    threshold : int
        Termination threshold
    """

    def __init__(
        self,
        tools,
        model=None,
        evaluator=None,
        prompt=AGENT_PROMPT,
        depth=5,
        b_factor=3,
        budget=None,  # TODO: Set a default budget
        threshold=0.7,
    ):
        # Setup
        self.tools = tools
        self.model = model if model else OpenAI()
        self.evaluator = evaluator
        self.prompt = prompt

        # Hyperparameters
        self.depth = depth
        self.b_factor = b_factor
        self.budget = budget
        self.threshold = threshold

        # Token limits
        self.tool_call_headroom = 1024
        self.eval_headroom = 512

        # Validations
        assert_valid_tools(tools)
        assert_valid_threshold(threshold)

    def get_tool_schemas(self):
        return [tool.get_schema() for tool in self.tools]

    def get_tool_by_name(self, name: str) -> Union[Tool, None]:
        for tool in self.tools:
            if tool.name == name:
                return tool

        return None

    def get_trimmed_messages(self, state: State, headroom: int = 1024):
        """
        Converts the state into a list of messages. If there are too many tokens
        in the messages, the oldest messages are trimmed until the token limit
        is no longer breached.
        """

        headroom = self.model.get_context_window() - headroom
        token_count = self.model.count_message_tokens(state.instruction)
        token_count += sum(self.model.count_tool_tokens(tool) for tool in self.tools)

        trimmed_messages = [state.instruction]
        for message in reversed(state):
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

    async def evaluate(self, state: State) -> float:
        """
        Evaluates the current branch of the search tree, i.e. a tool-use trajectory.
        The evaluation tells the agent how well the trajectory aligns with the given
        instruction.

        Returns a value between 0 and 1, where 1 is the best possible trajectory
        (i.e. the goal state).
        """

        # Initial state always has a score of 0
        if state.is_empty():
            return 0

        # User has provided a custom evaluator
        if self.evaluator:
            return self.evaluator.run(state)

        # Use default evaluator
        system_message = Message.system(EVAL_PROMPT)
        headroom = self.model.count_message_tokens(system_message) + self.eval_headroom
        messages = self.get_trimmed_messages(state, headroom)
        messages += [system_message]
        response = await self.model.run(
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
                            "score": {
                                "type": "number",
                                "description": "A score between 0 and 1 indicating how well the tool-use sequence aligns with the user's instruction.",
                            },
                            "reason": {
                                "type": "string",
                                "description": "A brief explanation of why you gave this score.",
                            },
                        },
                    },
                },
            },
        )
        response = json_repair.loads(response.content)
        return response["score"]

    async def sample_actions(self, state: State) -> List[Message]:
        """
        Generates plausible next actions to take in a given trajectory. Obtains _b_
        candidate actions by asking a language model.

        - For each tool, make a meta-tool with just one parameter: a list of the tool's
        parameter objects. Prompt it to always generate _b_ parameter objects.
        - In the system prompt, instruct the model to always generate _b_
        parallel tool calls.
        """

        # num_outputs=max(branching_factor * 2, 20), n completion param

        system_message = Message.system(self.prompt)
        headroom = (
            self.model.count_message_tokens(system_message) + self.tool_call_headroom
        )
        messages = self.get_trimmed_messages(state, headroom)
        messages += [system_message]
        response = await self.model.run(
            messages,
            tools=self.get_tool_schemas(),
            parallel_tool_calls=True,
            tool_choice="required",
            max_tokens=self.tool_call_headroom,
        )
        tool_calls = Message.from_openai_message(response)
        return tool_calls

    async def execute_action(self, state: State, action: Message) -> State:
        """
        Executes a tool call and appends the result to the trajectory.
        """

        # Execute the tool call
        fn_call = action.tool_calls[0]
        tool = self.get_tool_by_name(fn_call.name)
        output = await tool.run(**fn_call.arguments)
        formatted_output = tool.format_output(output)
        result = Message.tool(formatted_output, fn_call.id)

        # TODO: Store raw output somewhere
        # TODO: This doesn't handle parallel tool calls

        # Update the state
        new_state = state.from_state(state)
        new_state.add_message(action)
        new_state.add_message(result)

        return new_state

    async def run(self, instruction: str) -> State:
        """
        Performs a best-first tree search for the optimal tool-use trajectory.
        """

        initial_state = State(instruction)
        initial_score = self.evaluate(initial_state)

        # Max priority queue (negative scores for max behavior)
        frontier = []
        best_state = initial_state
        best_score = -inf
        s_counter = 0  # Search counter

        # Push the initial state to the frontier
        heapq.heappush(frontier, (-initial_score, best_state))

        # Best-first search
        while s_counter < self.budget:
            if not frontier:
                break  # No more states to explore

            # Get the next state to explore
            neg_score, curr_state = heapq.heappop(frontier)
            curr_score = -neg_score  # Convert back to positive score

            s_counter += 1

            # Update the best state if curr_score is better
            if curr_score > best_score:
                best_score = curr_score
                best_state = curr_state

            # Termination conditions
            if curr_score >= self.threshold or s_counter >= self.budget:
                break

            if len(curr_state) < self.depth:
                # Generate candidates for the next action
                actions = self.sample_actions(curr_state)

                # Execute and evaluate each action
                for action in actions:
                    next_state = self.execute_action(curr_state, action)
                    next_score = self.evaluate(next_state)
                    heapq.heappush(frontier, (-next_score, next_state))

        return best_state
