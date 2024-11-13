# Standard library
from typing import List, Optional

# Local
from src.dtos import Message, Node
from src.abstract import Tool, Model
from src.prompts import AGENT_PROMPT


class MonteCarloAgent(object):
    def __init__(
        self,
        tools: List[Tool],
        model: Optional[Model] = None,
        evaluator: Optional[any] = None,
        prompt: str = AGENT_PROMPT,
        b_factor: int = 3,
        max_depth: int = 5,
        threshold: float = 0.7,
        # MCTS hyperparameters
        max_rollouts: int = 10,
    ):
        super().__init__(
            tools, model, evaluator, prompt, b_factor, max_depth, threshold
        )
        self.max_rollouts = max_rollouts

    def should_terminate(self, tree: Node, num_sims: int) -> bool:
        if tree.is_solved:
            return True

        if tree.height >= self.max_depth:
            return True

        if num_sims >= self.max_rollouts:
            return True

        return False

    async def generate_root_node(self, prompt: str) -> Node:
        """
        Generates the root node (i.e. the first tool call) in the
        search tree.
        """

        # TODO: If this root tool call is wrong, then the whole search tree is screwed.
        # We should use the prompt as the root node and start the search by expanding that.

        # Generate the first tool call
        system_message = Message.system(self.prompt)
        user_message = Message.user(prompt)
        response = await self.model.arun(
            [system_message, user_message],
            tools=self.get_tool_schemas(),
            parallel_tool_calls=False,
            tool_choice="required",
            max_tokens=self.tool_call_headroom,
            temperature=1.0,
        )
        tool_call = Message.from_response(response)

        # Execute the tool call
        tool_response = await self.execute_tool_call(tool_call)

        # Build and evaluate the root node
        node = Node([Message.user(prompt), tool_call, tool_response])
        await self.evaluate(node)

        return node

    def has_non_terminal_leaves(self, root: Node) -> bool:
        leaf_nodes = root.get_leaf_nodes()
        return any(not self.is_terminal_node(node) for node in leaf_nodes)

    def select(self, root: Node) -> Optional[Node]:
        """
        Selects the best (leaf) node to expand using the UCB algorithm. If all paths in
        the tree have been exhausted, this returns `None`.
        """

        node = root
        while node and not node.is_leaf:
            non_terminal_children = (
                child for child in node.children if not self.is_terminal_node(child)
            )
            viable_children = (
                child
                for child in non_terminal_children
                if self.has_non_terminal_leaves(child) or child.is_leaf
            )  # Leaf nodes, or subtrees with non-terminal leaves
            node = max(
                viable_children,
                key=lambda child: child.upper_confidence_bound(),
                default=None,
            )

        return node

    async def simulate(self, node: Node):
        """
        Simulates a rollout from the given node until a terminal node is reached.
        If the terminal node is a solution node, then we mark the tree as solved.
        Otherwise, we backpropagate the score up the tree and a self-reflection.
        """

        curr_node = node.get_best_child()
        while not self.is_terminal_node(curr_node):
            await self.expand(curr_node)
            curr_node = curr_node.get_best_child()

        if self.is_solution_node(curr_node):
            curr_node.mark_as_solved()
            return

        # curr_node.self_reflect() # TODO
        curr_node.backpropagate()

    async def run(self, prompt: str) -> Node:
        num_sims = 0
        root = await self.generate_root_node(prompt)
        while not self.should_terminate(root, num_sims):
            node = self.select(root)
            await self.expand(node)
            await self.simulate(node)

            num_sims += 1

        return self.get_best_node(root)
