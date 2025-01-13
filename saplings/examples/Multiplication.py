# Local
try:
    from saplings.abstract import Tool
except ImportError:
    from abstract import Tool


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

    def format_output(self, output: any) -> str:
        return f"{output['a']} * {output['b']} = {output['result']}"

    async def run(self, a: int, b: int, **kwargs):
        return {"a": a, "b": b, "result": a * b}
