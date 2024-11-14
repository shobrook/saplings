# Standard library
import math
from collections import deque
from typing import Generator, List, Optional

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
        self.id = id(self)
        self.messages = messages  # Tool calls + responses
        self.parent = parent
        self.children = []
        self.depth = parent.depth + 1 if parent else 1
        self.visits = 0
        self._value = 0
        self.is_solved = False
        self.set_evaluation(evaluation)

    def __repr__(self):
        tab = "   "
        node_str = "Node(\n"
        node_str += f"{tab}id={self.id},\n"
        node_str += f"{tab}parent={self.parent.id if self.parent else -1},\n"
        node_str += f"{tab}depth={self.depth},\n"
        node_str += f"{tab}visits={self.visits},\n"
        node_str += f"{tab}value={self.value},\n"
        node_str += f"{tab}score={self.score}\n"
        node_str += f"{tab}state=[\n"
        for message in self.messages:
            for line in str(message).split("\n"):
                node_str += f"{tab}{tab}{line}\n"
        node_str += f"{tab}],\n"
        node_str += ")"

        return node_str

    def __str__(self):
        bold = "\033[1m"
        red = "\033[91m"
        yellow = "\033[93m"
        green = "\033[92m"
        reset = "\033[0m"

        value_color = red
        if self.value >= 0.33 and self.value < 0.67:
            value_color = yellow
        elif self.value >= 0.67:
            value_color = green

        node_str = f"{bold}===== NODE (depth={self.depth}) ====={reset}\n\n"
        node_str += "\n".join(str(message) for message in self.messages)
        node_str += f"\n{bold}VALUE:{reset} {value_color}{self.value}{reset}"
        node_str += f"\n\n{bold}===== END NODE ====={reset}"

        return node_str

    def __lt__(self, other):
        # NOTE: Used by heapq to compare nodes
        return self.score < other.score

    @property
    def score(self) -> float:
        """
        Returns the self-evaluation score of this node (float between 0 and 1).
        """

        return self.evaluation.score if self.evaluation else 0

    @property
    def value(self) -> float:
        """
        Returns the value of this node. For A* and BFS, this is equivalent to self.score.
        For MCTS, this is the score modified by backpropagation.
        """

        return self._value

    @property
    def height(self) -> int:
        """
        Returns the height of the tree (i.e. # of levels) rooted at this node.
        """

        if self.children:
            return 1 + max(child.height for child in self.children)

        return 1

    @property
    def is_leaf(self) -> bool:
        """
        Returns whether this node is a leaf node.
        """

        return not self.children

    def set_evaluation(self, evaluation: Optional[Evaluation]):
        # NOTE: We only need this method because we create nodes before they're
        # evaluated. This is just for convenience and can be confusing. Should
        # eventually refactor this.

        self.evaluation = evaluation
        self._value = evaluation.score if evaluation else 0

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

    def get_best_child(self) -> Optional["Node"]:
        if not self.children:
            return None

        return max(self.children, key=lambda child: child.value)

    def get_leaf_nodes(self) -> Generator["Node", None, None]:
        """
        Get all the leaf nodes rooted at this node.
        """

        nodes = deque([self])
        while nodes:
            node = nodes.popleft()
            if node.is_leaf:
                yield node
            else:
                nodes.extend(node.children)

    def bfs(self) -> Generator["Node", None, None]:
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

        # TODO: Double-check that division by self.visits is correct here
        exploitation_term = self.value / self.visits
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation_term + exploration_weight * exploration_term

    def mark_as_solved(self):
        """
        Marks this node as solved and then marks the whole tree
        as solved.
        """

        node = self
        while node:
            node.is_solved = True
            node = node.parent

    def backpropagate(self):
        """
        Updates the value of this node and its parents.
        """

        reward = self.value
        node = self
        while node:
            node.visits += 1
            node._value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

            # node.visits += node.parent.visits
            # node.value = (node.parent.value * node.parent.visits + reward) / node.visits
            # node. = node.parent
