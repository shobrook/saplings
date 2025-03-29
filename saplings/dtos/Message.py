# Standard library
from typing import List, Union

# Third party
import json_repair

# Local
try:
    from saplings.dtos.ToolCall import ToolCall
except ImportError:
    from dtos.ToolCall import ToolCall


class Message(object):
    def __init__(
        self,
        role: str,
        content: Union[None, str] = None,
        tool_calls: Union[None, List[ToolCall]] = None,
        tool_call_id: int = None,
        raw_output: any = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id

        # For tool messages (unformatted tool call output)
        self.raw_output = raw_output

        # Set after initialization
        self.score = None
        self.parent_id = None
        self.id = None

    @classmethod
    def system(cls, content):
        return Message("system", content)

    @classmethod
    def user(cls, content):
        return Message("user", content)

    @classmethod
    def assistant(cls, content):
        return Message("assistant", content)

    @classmethod
    def tool_calls(cls, tool_calls):
        return cls(role="assistant", tool_calls=tool_calls)

    @classmethod
    def tool(cls, content, id, raw_output=None):
        return cls(
            role="tool",
            content=content,
            tool_call_id=id,
            raw_output=raw_output,
        )

    @classmethod
    def from_openai_message(cls, message):
        role = message.role
        content = message.content

        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                args = json_repair.loads(tool_call.function.arguments)
                tool_call = ToolCall(tool_call.id, tool_call.function.name, args)
                tool_calls.append(tool_call)

            return cls.tool_calls(tool_calls)

        return cls(role, content=content)

    def to_openai_message(self):
        message = {"role": self.role, "content": self.content}
        if self.role == "tool":
            message["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            message["tool_calls"] = [
                tool_call.to_dict() for tool_call in self.tool_calls
            ]
        return message

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        bold = "\033[1m"
        grey = "\033[37m"
        reset = "\033[0m"

        if self.role == "tool":
            return f'{bold}TOOL OUTPUT:{reset} {grey}"{self.content}"{reset}'
        elif self.role == "user":
            return f'{bold}USER INPUT:{reset} {grey}"{self.content}"{reset}'
        elif self.role == "assistant":
            if self.tool_calls:
                tool_calls_str = ""
                for tool_call in self.tool_calls:
                    tool_calls_str += (
                        f"{bold}TOOL CALL:{reset} {grey}{str(tool_call)}{reset}\n"
                    )

                return tool_calls_str.strip()

            return f'{bold}ASSISTANT OUTPUT:{reset} {grey}"{self.content}"{reset}'
        elif self.role == "system":
            return f'{bold}SYSTEM MESSAGE:{reset} {grey}"{self.content}"{reset}'

        return ""

    def __hash__(self):
        tool_call_hashes = [hash(tc) for tc in self.tool_calls or []]
        return hash(
            (
                self.role,
                self.content,
                tuple(tool_call_hashes) or "",
            )
        )
