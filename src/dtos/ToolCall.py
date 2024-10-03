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

    def __repr__(self):
        return f"ToolCall({self.name}, {self.arguments})"
