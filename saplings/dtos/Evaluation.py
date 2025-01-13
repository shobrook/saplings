# Standard library
from typing import Optional

# Third party
import json_repair

# Local
try:
    from saplings.dtos.Message import Message
except ImportError:
    from dtos.Message import Message


class Evaluation(object):
    def __init__(self, score: int, reasoning: Optional[str] = None):
        self.score = score
        self.reasoning = reasoning

    def to_message(self) -> Message:
        return Message.user(f"Reasoning: {self.reasoning}\nScore: {self.score * 10}")

    @classmethod
    def from_message(cls, message: Message) -> "Evaluation":
        arguments = json_repair.loads(message.content)
        reasoning = arguments.get("reasoning", "")
        score = arguments.get("score", 5)
        score = max(0, min(score, 10))  # Ensures score is between 0 and 10
        score = score / 10.0  # Normalizes score to be between 0 and 1
        return cls(score, reasoning)
