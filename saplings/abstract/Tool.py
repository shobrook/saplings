# Standard library
from typing import List
from abc import ABC, abstractmethod

# Local
try:
    from saplings.dtos import Message
except ImportError:
    from dtos import Message


class Tool(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        self.name: str
        self.description: str
        self.parameters: dict
        self.is_terminal: bool

    @abstractmethod
    async def run(self, **kwargs) -> any:
        return None

    def is_active(self, trajectory: List[Message] = [], **kwargs) -> bool:
        return True

    def update_definition(self, trajectory: List[Message] = [], **kwargs) -> any:
        pass

    def format_output(self, output: any) -> str:
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


# TODO: Implement a `from_langchain_tool` static method?
