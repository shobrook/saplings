# Standard library
import asyncio

# Local
from src.dtos import Message, Node


class MonteCarloAgent(object):
    def __init__(self, height: int = 5):
        self.height = height

    def is_node_terminal(self, node: Node) -> bool:
        if node.is_solved:
            return True

        if node.depth >= self.depth:
            return True

        if self.is_output_node(node):
            return True

        return False

    async def generate_root_node(self, prompt: str) -> Node:
        """
        Generates the root node (i.e. the first tool call) in the
        search tree.
        """

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

    def select(self, root: Node) -> Node:
        """
        Selects the best (leaf) node to expand using the UCB algorithm.
        """

        # If it's a leaf node, we select it
        if not root.children:
            return root

        # Otherwise, we select the child node with the highest UCB
        node = root
        while node.children:
            max_child = max(
                node.children, key=lambda child: child.upper_confidence_bound()
            )
            node = max_child

        return node

    async def expand(self, node: Node) -> Node:
        """
        Generates candidates for the next action in the search tree.
        """

        # Generate candidate next tool calls, execute each
        tool_calls = await self.generate_candidates(node)
        tasks = [self.execute_tool_call(tool_call) for tool_call in tool_calls]
        tool_responses = await asyncio.gather(*tasks)

        # Create new nodes
        nodes = [
            Node([call, response], parent=node)
            for call, response in zip(tool_calls, tool_responses)
        ]

        # TODO: Question. We should be evaluating each child node and adding them
        # to the tree. But when we evaluate a node, we're also backpropagating the
        # score up the tree. Should we be doing this for EVERY child node? Or should 
        # we only backpropagate once a TERMINAL node is reached?

        # Evaluate each node
        tasks = [self.evaluate(node) for node in nodes]
        await asyncio.gather(*tasks)

        # Grow the tree
        node.add_children(nodes)

        return nodes

    def should_terminate(self, root: Node) -> bool:
        if root.is_solved:
            return True

        if root.height > self.height:
            return True

        return False

    async def run(self, prompt: str) -> Node:
        root = await self.generate_root_node(prompt)
        while not self.should_terminate(root): # Depth limit, iteration limit. If solved node then we return it which will break the loop.
            node = self.select(root) # Get best candidate for expansion
            await self.expand(node)
            node.get_best_child() #
        
        return root.get_best_solution()


root
node = select(root) -> root
children = expand(node)
best_child = max(children, key=lambda child: child.value) # When we evaluate a node, set node.value = score
simulate(best_child) # Rollout from best_child
    curr_node = best_child
  - while not self.is_terminal_node(curr_node): # Output node, depth limit, or solved node
       children = expand(curr_node)
       best_child = max(children, key=lambda child: child.value)
       curr_node = best_child

   if curr_node.is_solved:
        return curr_node
    else:
       curr_node.backpropagate()
    
    curr_node.self_reflect()

node = select(root)
children = expand(node)


# class MonteCarloAgent(object):
#     def __init__(self):
#         pass

#     async def run(self, prompt: str) -> Node:
#         root = await self.generate_root_node(prompt)
#         for _ in range(self.num_rollouts):
#             node = await self.select(root)
            
#             # All paths from the root are exhausted. Each leads to an unsolved terminal node.
#             if not node: 
#                 break

#             if self.is_terminal_node(node) and node.is_solved:
#                 return node
            
#             await self.expand(node)
            





s = initial state
p_theta = action generator
p_v = value function
p_ref = reflection generator
n = number of generated actions (b_factor)
L = depth limit
K = number of rollouts
c = context
w = exploration weight
lambda = value function weight
A = action space
O = observation space
N = visit counter


def select(root: Node) -> Node:
    """
    Selects the best (leaf) node in the tree to expand using the 
    UCT algorithm.
    """

    return

def run(self):
    root = self.generate_root_node()
    node = select(root)

    children = expand(node)
    for child in children:
        evaluate(child) # Reasoning + score (LLM). Then self-consistency. Then V = lambda * llm_score + (1 - lambda) * sc_score

    for _ in range(self.num_rollouts):
        while root.height <= self.depth:
            pass


    for k in range(self.num_rollouts):
        for t in range(self.depth):
            if not node.is_terminal:
                # Expand node
                self.expand(node) # Generates tool calls, executes each, evaluates each child node
            else: 
                # Terminal node. Either solution_node or is_solved (score >= threshold).

                if not node.is_solved:
                    # Generate reflection
                    # Stuff trajectory + evaluation into prompt to generate self-reflection
                    # - Should summarize errors in the reasoning or acting process and propose superior alternatives
                    # - Used as context for the agent and value function
                    node.context = reflect(node)


# Evaluator should return a score and "is_solved". If
# not output node (and self.can_self_terminate), then is_solved = False
# If not output node (and not self.can_self_terminate), then is_solved = True if score > threshold.




# Can call terminal tools "Output actions".

def select_node(node):
    while node and node.children:
        logging.info(f"Selecting from {len(node.children)} children at depth {node.depth}.")
        
        terminal_children = [child for child in node.children if child.is_terminal]
        terminal_status = [child.is_terminal for child in node.children]
        
        if len(terminal_children) == len(node.children):
            logging.info(f"All children are terminal at depth {node.depth}. Backtracking...")
            if node.parent:  
                node.parent.children.remove(node)
            node = node.parent  
            continue  
        
        node_with_reward_1 = next((child for child in terminal_children if child.reward == 1), None)
        if node_with_reward_1:
            logging.info(f"Found terminal node with reward 1 at depth {node.depth}.")
            return node_with_reward_1
        
        node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)

        while node.is_terminal and node.reward != 1:
            node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
            
        logging.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")
        
    return node  # This will return None if all paths from the root are exhausted
