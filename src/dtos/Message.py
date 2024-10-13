# Standard library
import json
from typing import List, Union

# Third party
import json_repair

# Local
from app.shared.dtos.ToolCall import ToolCall


class Message(object):
    def __init__(
        self,
        role: str,
        content: Union[None, str] = None,
        tool_calls: Union[None, List[ToolCall]] = None,
        tool_call_id: int = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id

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
    def tool(cls, content, id):
        return cls(role="tool", content=content, tool_call_id=id)

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
        return f"Message(role={self.role}, content={self.content}, tool_calls={str(self.tool_calls)})"