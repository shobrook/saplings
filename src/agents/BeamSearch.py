# Standard library
import asyncio
from math import mean
from typing import List, Union
from collections import defaultdict

# Third party
import json_repair

# Local
from src.llms import OpenAI
from src.abstract import Tool
from src.dtos import State, Message
from src.prompts import EVAL_PROMPT, AGENT_PROMPT


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


class BeamSearchAgent(object):
    def __init__(
        self,
        tools,
        model=None,
        evaluator=None,
        prompt=AGENT_PROMPT,
        depth=5,
        b_factor=3,
        budget=15,
        threshold=0.7,
    ):
        # Setup
        self.tools = tools
        self.model = model if model else OpenAI()
        self.evaluator = evaluator
        self.prompt = prompt

        # Hyperparameters
        self.depth = depth  # Search depth
        self.b_factor = b_factor  # Branching factor
        self.budget = budget  # Maximum size of the search tree
        self.threshold = threshold  # Termination threshold

        # Token limits
        self.tool_call_headroom = 1024 * b_factor  # TODO: Necessary?
        self.eval_headroom = 256

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
            n=20,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Your thoughts and reasoning process. Keep it brief and concise.",
                            },
                            "is_success": {
                                "type": "boolean",
                                "description": "Whether the agent has succeeded in satisfying the user's intent.",
                            },
                            "is_right": {
                                "type": "boolean",
                                "description": "Whether the agent is on the right track to success.",
                            },
                        },
                    },
                },
            },
        )

        # Calculate score
        scores = []
        for choice in response:
            score = json_repair.loads(choice.message.content)
            if score["is_success"]:
                scores.append(1.0)
            elif score["is_right"]:
                scores.append(0.5)
            else:
                scores.append(0.0)

        return mean(scores)

    async def sample_actions(self, state: State) -> List[Message]:
        """
        Generates plausible next actions to take in a given trajectory (state).
        Obtains *b_factor* candidate actions by asking a language model.

        Two approaches here, since we're a function-calling agent and not a
        regular ReAct agent:
        1. Rely on enabling parallel tool calls to generate *b_factor* tool calls
           - Requires specific instructions for this in the system prompt
           - Not guaranteed to always produce the minimum # of tool calls
        2. Use the `num_outputs` parameter to control the number of tool calls
           - Guaranteed to produce the minimum # of tool calls
           - Downside? Possibly not enough variation between tool calls

        We are using approach #2 for now.
        """

        # Calculate # of tool calls to generate (minimum of 20)
        num_tools = len(self.tools)
        num_outputs = max(self.b_factor * num_tools, 20)

        # Generate tool calls
        system_message = Message.system(self.prompt)
        headroom = (
            self.model.count_message_tokens(system_message) + self.tool_call_headroom
        )
        messages = self.get_trimmed_messages(state, headroom)
        messages += [system_message]
        response = await self.model.run(
            messages,
            tools=self.get_tool_schemas(),
            parallel_tool_calls=False,
            tool_choice="required",
            max_tokens=self.tool_call_headroom,
            n=num_outputs,
            temperature=1.0,
        )

        # Get the top *b_factor* tool calls
        tool_counts = defaultdict(lambda: 0)
        tool_messages = defaultdict(dict)
        for choice in response:
            message = Message.from_openai_message(choice.message)
            tool_call = message.tool_calls[0]
            tool_call_id = f"{tool_call.name}:{tool_call.function.arguments}"

            tool_counts[tool_call_id] += 1
            tool_messages[tool_call_id] = message

        # TODO: Sort by tool name as tiebreaker
        top_tools = sorted(tool_counts.items(), key=lambda item: item[1], reverse=True)
        messages = [
            tool_messages[tool_call_id]
            for tool_call_id, _ in top_tools[: self.b_factor]
        ]

        return messages

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

        # Update the state
        new_state = state.from_state(state)
        new_state.add_message(action)
        new_state.add_message(result)

        return new_state

    async def run(self, instruction: str) -> State:
        """
        Performs a beam search for the optimal tool-use trajectory.
        """

        initial_state = State(instruction)
        initial_score = await self.evaluate(initial_state)

        beam_width = self.b_factor  # Beam width is set to the branching factor
        max_depth = self.depth  # Maximum search depth

        # Initialize the beam with the initial state
        beam = [(initial_score, initial_state)]
        best_state = initial_state
        best_score = initial_score

        for _ in range(max_depth):
            all_candidates = []

            # Expand each state in the current beam
            for score, state in beam:
                # Termination condition
                if score >= self.threshold:
                    return state  # Early exit if threshold is met

                # Generate candidate actions from the current state
                actions = await self.sample_actions(state)

                # Execute each action to get new states
                tasks = [self.execute_action(state, action) for action in actions]
                new_states = await asyncio.gather(*tasks)

                # Evaluate the new states
                tasks = [self.evaluate(new_state) for new_state in new_states]
                new_scores = await asyncio.gather(*tasks)

                # Collect all candidates
                all_candidates.extend(zip(new_scores, new_states))

            if not all_candidates:
                break  # No more candidates to expand

            # Select the top candidates to form the new beam
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beam = all_candidates[:beam_width]

            # Update the best state found so far
            if beam[0][0] > best_score:
                best_score, best_state = beam[0]

        return best_state
