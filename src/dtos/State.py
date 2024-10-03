# Local
from src.dtos.Message import Message


class State(object):
    """
    Represents the current state of a branch in the search tree. Contains the
    original instruction and a list of tool-use messages.
    """

    def __init__(self, instruction: str, messages=[]):
        self.instruction = Message.user(instruction)
        self.messages = messages
        self.score = None

    def is_empty(self) -> bool:
        return len(self.messages) == 0

    def add_message(self, message: Message):
        self.messages.append(message)
        self.is_empty = False

    def __len__(self):
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)

    @staticmethod
    def from_state(state):
        return State(state.instruction.content, state.messages)
