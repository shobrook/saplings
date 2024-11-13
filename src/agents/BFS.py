# Local
from src.agents.Base import BaseAgent
from src.dtos import Node


class BFSAgent(BaseAgent):
    def __init__(**kwargs):
        super().__init__(**kwargs)

    async def run(self, prompt: str) -> Node:
        pass
