# Standard library
from typing import List, Optional, Tuple

# Local
try:
    from saplings.model import Model
    from saplings.agents.Base import BaseAgent
    from saplings.dtos import Message, Node
    from saplings.abstract import Tool, Evaluator
    from saplings.prompts import AGENT_PROMPT
except ImportError:
    from model import Model
    from agents.Base import BaseAgent
    from dtos import Message, Node
    from abstract import Tool, Evaluator
    from prompts import AGENT_PROMPT


class MonteCarloAgent(BaseAgent):
    def __init__(
        self,
        tools: List[Tool],
        model: Optional[Model] = None,
        evaluator: Optional[Evaluator] = None,
        prompt: str = AGENT_PROMPT,
        b_factor: int = 3,
        max_depth: int = 5,
        threshold: float = 1.0,
        max_rollouts: int = 10,
        verbose: bool = True,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = False,
        update_prompt: Optional[callable] = None,
    ):
        super().__init__(
            tools,
            model,
            evaluator,
            prompt,
            b_factor,
            max_depth,
            threshold,
            verbose,
            tool_choice,
            parallel_tool_calls,
            update_prompt,
        )
        self.max_rollouts = max_rollouts

    def should_terminate(self, tree: Node, num_rollouts: int) -> bool:
        if tree.is_solved:
            return True

        if num_rollouts >= self.max_rollouts:
            return True

        return False

    async def generate_root_node(self, prompt: str, messages: List[Message]):
        """
        Generates the root node (i.e. the first tool call) in the
        search tree.
        """

        # TODO: If this root tool call is wrong, then the whole search tree is screwed.
        # We should use the prompt as the root node and start the search by expanding that.

        # Get active tools
        tools = [tool for tool in self.tools if tool.is_active(messages)]
        tool_schemas = [tool.get_schema() for tool in tools]

        # Generate the first tool call
        system_message = Message.system(self.prompt)
        user_message = Message.user(prompt)
        response = await self.model.run_async(
            [system_message] + messages + [user_message],
            tools=tool_schemas,
            parallel_tool_calls=False,
            tool_choice="required",
            max_tokens=self.max_tool_call_tokens,
            temperature=1.0,
        )
        tool_call = Message.from_openai_message(response)

        # Initialize the Node
        node = Node([Message.user(prompt), tool_call])

        tool_call.id = node.id
        yield tool_call

        # Execute the tool call
        tool_response = await self.execute_tool_call(
            tool_call, trajectory=[user_message]
        )
        node.messages.append(tool_response)

        # Build and evaluate the root node
        await self.evaluate(node)

        tool_response.id = node.id
        tool_response.score = node.score

        yield tool_response
        yield node

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

    async def simulate(self, node: Node, messages: List[Message] = []):
        """
        Simulates a rollout from the given node until a terminal node is reached.
        If the terminal node is a solution node, then we mark the tree as solved.
        Otherwise, we backpropagate the score up the tree and a self-reflection.
        """

        curr_node = node.get_best_child()
        while not self.is_terminal_node(curr_node):
            async for item in self.expand(curr_node, messages):
                yield item
            curr_node = curr_node.get_best_child()

        self.log(f"\033[1;31mReached terminal node\033[0m\n\n{curr_node}\n")

        if self.is_solution_node(curr_node):
            curr_node.mark_as_solved()
        else:
            # curr_node.self_reflect() # TODO
            curr_node.backpropagate()

    async def run_iter_async(self, prompt: str, messages: list[Message] = []):
        self.log(f"Running a Monte Carlo tree search\n\n\033[37m{prompt}\033[0m\n")

        root = None
        async for item in self.generate_root_node(prompt, messages):
            root = item
            yield item

        num_rollouts = 0
        while not self.should_terminate(root, num_rollouts):
            node = self.select(root)
            if not node:  # All paths exhausted
                break

            self.log(f"STARTING ROLLOUT (rollout_id={num_rollouts})")

            async for item in self.expand(node, messages):
                yield item

            async for item in self.simulate(node):
                yield item

            self.log(f"FINISHED ROLLOUT (rollout_id={num_rollouts})")

            num_rollouts += 1

        if root.is_solved:
            self.log("\033[1;32mFound a solution! Terminating search.\033[0m")
        else:
            self.log(
                "\033[1;31mNo solution found. Returning the best trajectory available.\033[0m"
            )

        best_node = self.get_best_node(root)
        messages, score, is_solution = (
            best_node.get_trajectory(),
            best_node.score,
            self.is_solution_node(best_node),
        )

        self.log(
            f"\033[1;32mBest trajectory (score={score}, is_solution={is_solution}):\033[0m\n\n"
            + "\n".join(str(m) for m in messages)
        )

        yield (messages, score, is_solution)
