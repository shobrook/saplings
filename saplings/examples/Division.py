# Local
try:
    from saplings.abstract import Tool
except ImportError:
    from abstract import Tool


class DivisionTool(Tool):
    def __init__(self, **kwargs):
        self.name = "divide"
        self.description = "Divides two numbers and returns the result number."
        self.parameters = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The numerator.",
                },
                "b": {"type": "number", "description": "The denominator."},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        }
        self.is_terminal = False

    async def run(self, a: int, b: int, **kwargs):
        return a / b
