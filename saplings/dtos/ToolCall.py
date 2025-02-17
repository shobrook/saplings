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
        return self.__str__()

    def __str__(self):
        return f"{self.name}({self.arguments})"
