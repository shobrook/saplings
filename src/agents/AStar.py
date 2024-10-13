# Standard library
import heapq
import asyncio
from math import inf

# Local
from src.dtos import State
from src.agents.Base import BaseAgent


class AStarAgent(BaseAgent):
    def __init__(**kwargs):
        super().__init__(**kwargs)

    async def run(self, instruction: str) -> State:
        """
        Performs an A* (best-first) search for the optimal tool-use trajectory.
        """

        initial_state = State(instruction)
        initial_score = await self.evaluate(initial_state)

        # Max priority queue (negative scores for max behavior)
        frontier = []
        best_state = initial_state
        best_score = -inf
        s_counter = 0  # Search counter

        # Push the initial state to the frontier
        heapq.heappush(frontier, (-initial_score, best_state))

        # Best-first search
        while s_counter < self.budget:
            if not frontier:
                break  # No more states to explore

            # Get the next state to explore
            neg_score, curr_state = heapq.heappop(frontier)
            curr_score = -neg_score  # Convert back to positive score

            s_counter += 1

            # Update the best state if curr_score is better
            if curr_score > best_score:
                best_score = curr_score
                best_state = curr_state

            # Termination conditions
            if curr_score >= self.threshold or s_counter >= self.budget:
                break

            if len(curr_state) < self.depth:
                # Generate candidates for the next action
                actions = await self.sample_actions(curr_state)

                # Execute each action candidate
                tasks = [self.execute_action(curr_state, action) for action in actions]
                states = await asyncio.gather(*tasks)

                # Evaluate each resulting state
                tasks = [self.evaluate(state) for state in states]
                scores = await asyncio.gather(*tasks)

                # Push the resulting states to the frontier
                for score, state in zip(scores, states):
                    heapq.heappush(frontier, (-score, state))

        return best_state
