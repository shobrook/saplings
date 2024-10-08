# Standard library
from abc import ABC, abstractmethod


class Tool(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        self.name: str
        self.description: str
        self.parameters: dict

    @abstractmethod
    async def run(self, **kwargs) -> any:
        return None

    @abstractmethod
    async def format_output(self, output: any) -> str:
        return str(output)

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": self.parameters,
            },
        }


# TODO: Implement a `from_langchain_tool` static method
# TODO: Add a return_direct field. If True, then when this
# tool is called, the agent cannot call additional actions
# as children. This tool is terminal.
