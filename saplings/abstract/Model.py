# Standard library
from typing import List
from abc import ABC, abstractmethod

# Local
try:
    from saplings.abstract import Tool
    from saplings.dtos import Message, ToolCall
except ImportError:
    from abstract.Tool import Tool
    from dtos import Message, ToolCall


class Model(ABC):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def get_context_window(self) -> int:
        return -1

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        return -1

    @abstractmethod
    def count_tool_call_tokens(self, tool_call: ToolCall) -> int:
        return -1

    @abstractmethod
    def count_message_tokens(self, message: Message) -> int:
        return -1

    @abstractmethod
    def count_tool_tokens(self, tool: Tool) -> int:
        return -1

    @abstractmethod
    def run(self, **kwargs) -> any:
        return None

    @abstractmethod
    async def run_async(self, **kwargs) -> any:
        return None

    def truncate_messages(
        self, messages: List[Message], headroom: int, tools: List[Tool] = []
    ) -> List[Message]:
        """
        Trims + drops messages to make room for the output headroom.

        Rules:
        1. The first message (user input) is always kept.
        2. We drop older messages before newer ones.
        3. We drop tool output before tool calls.
        4. We drop evaluation messages before tool calls/outputs. (TODO)
        """

        input_message = messages[0]
        headroom = self.get_context_window() - headroom
        token_count = self.count_message_tokens(input_message)
        token_count += sum(self.count_tool_tokens(tool) for tool in tools)

        truncated_messages = [input_message]
        for message in reversed(messages[1:]):
            num_tokens = self.count_message_tokens(message)
            if token_count + num_tokens > headroom:
                if message.role == "tool":
                    message.content = "[HIDDEN]"
                    num_tokens = self.count_message_tokens(message)

                    if token_count + num_tokens <= headroom:
                        token_count += num_tokens
                        truncated_messages.insert(1, message)
                        continue

                break

            token_count += num_tokens
            truncated_messages.insert(1, message)

        return truncated_messages
