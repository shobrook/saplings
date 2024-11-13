# Standard library
import json


class ToolCall(object):
    def __init__(self, id, name, arguments):
        self.id = id
        self.name = name
        self.arguments = arguments

    def to_dict(self):
        return {
            "id": self.id,
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
            "type": "function",
        }

    def __hash__(self):
        sorted_arguments = json.dumps(dict(sorted(self.arguments.items())))
        return hash((self.name, sorted_arguments))

    def __repr__(self):
        tab = "   "
        tool_call_str = "ToolCall(\n"
        tool_call_str += f"{tab}id={self.id},\n"
        tool_call_str += f"{tab}name={self.name},\n"
        tool_call_str += f"{tab}arguments={self.arguments}\n"
        tool_call_str += ")"

        return tool_call_str

    def __str__(self):
        return self.__repr__()
