# Local
from src.dtos.Message import Message


class Evaluation(object):
    def __init__(self, score: int, reasoning: str):
        self.score = score
        self.reasoning = reasoning

    def to_message(self) -> Message:
        return Message.user(f"Reasoning: {self.reasoning}\nScore: {self.score}")

    @classmethod
    def from_message(cls, message: Message) -> "Evaluation":
        arguments = message.tool_calls[0].arguments
        reasoning = arguments.get("reasoning", "")
        score = arguments.get("score", 5)
        score = max(0, min(score, 10))  # Ensures score is between 0 and 10
        score = score / 10.0  # Normalizes score to be between 0 and 1
        return cls(score, reasoning)
