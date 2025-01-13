# Standard library
from typing import List
from abc import ABC, abstractmethod

# Local
try:
    from saplings.dtos import Message, Evaluation as EvaluationDTO
except ImportError:
    from dtos import Message, Evaluation as EvaluationDTO


class Evaluator(ABC):
    @abstractmethod
    async def run(self, trajectory: List[Message]) -> EvaluationDTO:
        return None
