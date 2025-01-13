# Local
try:
    from saplings.abstract import Tool
except ImportError:
    from abstract import Tool


class AdditionTool(Tool):
    def __init__(self, **kwargs):
        self.name = "add"
        self.description = "Adds two numbers and returns the result number."
        self.parameters = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number to add.",
                },
                "b": {"type": "number", "description": "The second number to add."},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        }
        self.is_terminal = False

    async def run(self, a: int, b: int, **kwargs):
        return a + b
