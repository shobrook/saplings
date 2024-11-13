# Local
from src.abstract import Tool


class MultiplicationTool(Tool):
    def __init__(self, **kwargs):
        self.name = "multiply"
        self.description = "Multiplies two numbers and returns the result number."
        self.parameters = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The number to multiply.",
                },
                "b": {"type": "number", "description": "The number to multiply by."},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        }
        self.is_terminal = False

    async def run(self, a: int, b: int, **kwargs):
        return a * b
