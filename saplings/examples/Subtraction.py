# Local
try:
    from saplings.abstract import Tool
except ImportError:
    from abstract import Tool


class SubtractionTool(Tool):
    def __init__(self, **kwargs):
        self.name = "subtract"
        self.description = "Subtracts two numbers and returns the result number."
        self.parameters = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The number to subtract from.",
                },
                "b": {
                    "type": "number",
                    "description": "The number you're subtracting.",
                },
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        }
        self.is_terminal = False

    async def run(self, a: int, b: int, **kwargs):
        return a - b
