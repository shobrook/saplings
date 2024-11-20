# Standard library
from typing import List
from abc import ABC, abstractmethod

# Local
from saplings.dtos import Message, Evaluation as EvaluationDTO


class Evaluator(ABC):
    @abstractmethod
    async def run(self, trajectory: List[Message]) -> EvaluationDTO:
        return None
