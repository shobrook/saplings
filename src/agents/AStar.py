# Standard library
import heapq
import asyncio
from math import inf
from typing import List, Optional

# Local
from src.abstract import Tool, Model
from src.agents.Base import BaseAgent
from src.dtos import Message, Node
from src.prompts import AGENT_PROMPT


class AStarAgent(BaseAgent):
    def __init__(
        self,
        tools: List[Tool],
        model: Optional[Model] = None,
        evaluator: Optional[any] = None,
        prompt: str = AGENT_PROMPT,
        # A* hyperparameters
        b_factor: int = 3,
        depth: int = 5,
        budget: int = 15,
        threshold: float = 0.7,
    ):
        super().__init__(tools, model, evaluator, prompt, b_factor)
        self.depth = depth  # Search depth
        self.budget = budget  # Maximum size of the search tree
        self.threshold = threshold  # Termination threshold

    async def expand(self, node: Node) -> List[Node]:
        if self.is_output_node(node):
            return []

        # Generate candidates for the next action, execute each
        num_candidates = max(self.b_factor * len(self.tools), 20)
        tool_calls = await self.generate_candidates(node, num_candidates)
        tasks = [self.execute_tool_call(tool_call) for tool_call in tool_calls]
        tool_responses = await asyncio.gather(*tasks)

        # Create new nodes
        nodes = [
            Node([call, response], parent=node)
            for call, response in zip(tool_calls, tool_responses)
        ]

        # TODO: Double-check that order is preserved before zipping

        # Evaluate each new node
        tasks = [self.evaluate(node) for node in nodes]
        await asyncio.gather(*tasks)

        # Grow the tree
        node.add_children(nodes)

        return nodes

    def should_terminate(self, node: Node, s_counter: int) -> bool:
        """
        Determines if the search should be terminated. It should if:

        1. The maximum # of nodes has been explored.
        2. The current node's score is above the solution threshold AND the agent
           cannot self-terminate.
        3. The current node's score is above the solution threshold AND the node is
           a solution (terminal) node.
        """

        if s_counter >= self.budget:
            return True

        if node.normalized_score >= self.threshold:
            if not self.agent_has_output_tools():
                return True

            if self.is_output_node(node):
                return True

        return False

    async def run(self, prompt: str) -> Node:
        """
        Performs an A* (best-first) search for the optimal tool-use trajectory. Returns
        the leaf node in the optimal search branch, from which the entire trajectory
        can be reconstructed.
        """

        # Max priority queue (negative scores for max behavior)
        root_node = Node([Message.user(prompt)])
        best_score = -inf
        frontier = []

        # Push the initial node to the frontier
        heapq.heappush(frontier, (0, root_node))

        # Begin the search
        s_counter = 0
        while s_counter < self.budget:
            # No more nodes to explore
            if not frontier:
                break

            # Get the next node to explore
            neg_score, curr_node = heapq.heappop(frontier)
            curr_score = -neg_score  # Convert back to positive score

            s_counter += 1

            # Update the best node if curr_score is better
            if curr_score > best_score:
                best_score = curr_score

            if self.should_terminate(curr_node, s_counter):
                break

            if curr_node.depth < self.depth:
                # Expand the current node
                children = await self.expand(curr_node)

                # Push the resulting nodes to the frontier
                for child in children:
                    heapq.heappush(
                        frontier, (-child.evaluation.normalized_score, child)
                    )

        return self.get_best_node(root_node)
