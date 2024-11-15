# Standard library
from typing import List
from abc import ABC, abstractmethod

# Local
from src.dtos import Message, Evaluation as EvaluationDTO


class Evaluator(ABC):
    @abstractmethod
    async def run_async(self, trajectory: List[Message]) -> EvaluationDTO:
        return None
