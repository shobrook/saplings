# TreeAct

Plug-and-play implementation of tree search for LLM agents. Uses a variation of the A\* algorithm to find the optimal tool-use trajectory.

ReAct agents don't work well because they're vulnerable to compounding errors. Even a small mistake early in the loop can snowball and ruin the final output. `TreeAct` gives your agent backtracking abilities, making it easier to recover from such mistakes.

## Installation

Install the library using `pip`:

```bash
$ pip install TreeAct
```

## Quickstart

Using `TreeAct` is similar to using the `ReAct` agent in LangChain. All you need is some tools to get started.

```python
from treeact import TreeActAgent, Evaluator, DEFAULT_PROMPT
from treeact.llms import OpenAI
from treeact.examples import CalculatorTool

model = OpenAI()
tools = [CalculatorTool()] # Your tools
prompt = DEFAULT_PROMPT # Governs tool selection
evaluator = Evaluator(model) # Evaluates tool-use trajectories

agent = TreeActAgent(model, tools, prompt, evaluator)
agent.run("Do my taxes")
```

`Trajectory` object as input to the evaluator.

## Advanced usage

`TreeAct` gives you a lot of control over every component and step in the system. You can plug in your own LLM, evaluator, governing prompt, etc. Here are some options:

### Short-term Memory

### Evaluator (value function)

### ReAct style

OpenAI function calls vs. thought-action-observation.

### Custom prompt

For the ReAct prompt, use a stop token of "Observation:" to avoid hallucinations.

### Memory

Tools can return dictionaries or strings. All the tool calls are logged in a Memory object. This object is available to every tool, so if your tool wants to make use of specific output of a previous tool, you'll be able to do that.
