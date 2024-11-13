# Standard library
import math
from collections import deque
from typing import List, Optional

# Local
from src.dtos.Message import Message
from src.dtos.Evaluation import Evaluation


class Node(object):
    def __init__(
        self,
        messages: List[Message],
        evaluation: Optional[Evaluation] = None,
        parent: Optional["Node"] = None,
    ):
        self.messages = messages  # Tool calls + responses
        self.parent = parent
        self.children = []

        self.value = 0
        self.visits = 0
        self.depth = parent.depth + 1 if parent else 1
        self.set_evaluation(evaluation)

    # @property
    # def best_child_score(self) -> Optional["Node"]:
    #     """
    #     Return the child with the highest value.
    #     """

    #     if not self.children:
    #         return None

    #     return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """
        Check for how far we've rolled out the tree.
        """

        if self.children:
            return 1 + max(child.height for child in self.children)

        return 1

    @property
    def normalized_score(self) -> float:
        return self.evaluation.normalized_score if self.evaluation else 0

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def _mark_tree_as_solved(self):
        if not self.is_solved:
            return

        parent = self.parent
        while parent:
            parent.is_solved = True
            parent = parent.parent

    def set_evaluation(self, evaluation: Evaluation):
        self.evaluation = evaluation
        self.is_solved = evaluation.is_solved
        self._mark_tree_as_solved()
        self.backpropagate(evaluation.normalized_score)

    def get_messages(self, include_evals: bool = False) -> List[Message]:
        """
        Get all the messages represented by this node. I.e. the tool call(s),
        tool response(s), and (optionally) the model's self-evaluation of the
        node.
        """

        if include_evals and self.evaluation:
            return self.messages + [self.evaluation.to_message()]

        return self.messages

    def get_trajectory(self, include_evals: bool = False) -> List[Message]:
        """
        Get all the messages in this search branch.
        """

        messages = []
        node = self
        while node:
            messages += node.get_messages(include_evals)[::-1]
            node = node.parent

        messages = messages[::-1]
        return messages

    def add_children(self, children: List["Node"]):
        self.children.extend(children)

    def bfs(self):
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            yield node
            for n in node.children:
                nodes.append(n)

    def upper_confidence_bound(self, exploration_weight=1.0):
        """
        Calculates the UCT score. Helps balance exploration vs. exploitation of a branch.
        """

        if not self.parent:
            raise Exception("Root node has no parent")

        if self.visits == 0:
            return self.value

        # TODO: Should we divide by self.visits here?
        exploitation_term = self.value / self.visits
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation_term + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """
        Updates the score of this node and its parents.
        """

        reward = node.value
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

            # TODO: This might be the correct way to backpropagate the score
            # if node.parent:
            #     node.visits += node.parent.visits
            #     node.value = (node.parent.value * node.parent.visits + reward) / node.visits
