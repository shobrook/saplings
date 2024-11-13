# Local
from src.dtos.Message import Message


class Evaluation(object):
    def __init__(self, score: int, reasoning: str, is_solved: bool):
        self.score = score
        self.reasoning = reasoning
        self.is_solved = is_solved

    def to_message(self) -> Message:
        return Message.user(f"Reasoning: {self.reasoning}\nScore: {self.score}")

    @classmethod
    def from_message(cls, message: Message) -> "Evaluation":
        arguments = message.tool_calls[0].arguments
        reasoning = arguments.get("reasoning", "")
        score = arguments.get("score", 0)
        is_solved = arguments.get("is_solved", False)
        return cls(score, reasoning, is_solved)

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0
