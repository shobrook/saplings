# Local
from src.agents.Base import BaseAgent
from src.dtos import State


class GreedyAgent(BaseAgent):
    def __init__(**kwargs):
        super().__init__(**kwargs)

    async def run(self, instruction: str) -> State:
        pass
