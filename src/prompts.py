AGENT_PROMPT = """Your job is to choose the best action."""

EVAL_PROMPT = """You are an expert in evaluating the performance of an AI agent. The agent is designed to use tools to satisfy a user's intent. Given the user's intent and the agent's action history, your goal is to decide whether the agent is on the right track towards success. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.

Consider the tools used by the agent and their output. Evaluate the sequence of tool uses with respect to the given intent.

You MUST indicate whether you think the agent has FAILED or SUCCEEDED at satisfying the intent. And indicate whether the agent is on the right track, if it has failed.
"""

# Thoughts: <your thoughts and reasoning process>
# Status: "success" or "failure"
# On the right track to success: "yes" or "no"
