<div align="center">
  <img width="308" src="./logo.png" />
</div>

---

**Saplings is a plug-and-play framework for building agents that use search algorithms to complete tasks.**

By incorporating search, an agent can explore different tool-use trajectories and find the optimal path. This ability to look multiple steps ahead reduces errors and boosts overall task performance –– especially on complex reasoning problems, like code generation or navigating a website. With saplings, you can build search into your agents with just a couple lines of code.

- Supports popular search algorithms: **Monte Carlo Tree Search (MCTS), A\*, and greedy best-first search**
- Uses OpenAI function calling under the hood
- Full control over the evaluation function, prompts, search parameters, etc.

![Demo](./demo.png)

**Why add search?**

Chain-of-thought/ReAct-style agents don't work well because they're vulnerable to compounding errors. Even a small mistake early in the loop can snowball and ruin the final output. Adding tree search gives your agent lookahead and backtracking abilities, making it easier to recover from such mistakes. And as compute becomes cheaper, it will become table stakes for agents to use inference-time search.

---

- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Creating a tool](#creating-a-tool)
  - [Configuring an agent](#configuring-an-agent)
- [Docs](#docs)
  - [Agents](#agents)
    - [Parameters](#parameters)
    - [MonteCarloAgent: Monte Carlo tree search](#monte-carlo-tree-search)
    - [AStarAgent: A\* search](#a*-agent)
    - [GreedyAgent: Greedy best-first search](#greedy-best-first-search)
    - [ReActAgent: Chain-of-thought (no search)](#chain-of-thought)
  - [The `Message` object](#the-message-object)
  - [Advanced tool options](#advanced-tool-options)
    - [Accessing agent memory](#accessing-agent-memory)
    - [The `format_output()` method](<#the-format_output()-method>)
    - [Terminal tools](#terminal-tools)
  - [Custom evaluators](#custom-evaluators)
- [Roadmap](#roadmap)

## Installation

```bash
$ pip install saplings
```

## Quickstart

Below is a simple agent equipped with a tool for multiplying numbers together. It uses Monte Carlo Tree Search (MCTS) under the hood to solve tricky arithmetic problems.

```python
from saplings.examples import MultiplicationTool
from saplings.llms import OpenAI
from saplings import AStarAgent, Evaluator

model = OpenAI(model="gpt-4o", api_key="YOUR_API_KEY")
evaluator = Evaluator(model)
tools = [MultiplicationTool()]

agent = MonteCarloAgent(tools, model, evaluator)
messages, _, _ = agent.run("Let x = 9418.343 * 8.11 and y = 2x. Calculate (xy)(x^2).")
```

This is like the "bare minimum" for setting up a search agent with saplings –– there are a lot more parameters you can control, which is covered in the [docs](#docs). But let's first walk through how to create your own tools and configure an agent.

### Creating a tool

Tools are what your agent will use to perform a task or answer a query. Each tool must extend the `Tool` base class and implement a few variables and methods. In our example, we'll make a simple tool that multiplies two numbers together:

```python
from saplings.abstract import Tool

class MultiplicationTool(Tool):
   def __init__(self, **kwargs):
      self.name = "multiply"
      self.description = "Multiplies two numbers and returns the result number."
      self.parameters = {
         "type": "object",
         "properties": {
            "a": {
               "type": "number",
               "description": "The number to multiply."
            },
            "b": {
               "type": "number",
               "description": "The number to multiply by."
            }
         },
         "required": ["a", "b"],
         "additionalProperties": False
      }
      self.is_terminal = False

   async def run(self, a, b, **kwargs):
      return a * b
```

**Variables:**

These tell the model when and how to call your tool. If you've used [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling) before, this should be familiar to you.

- `name` (str): Name of the tool.
- `description` (str): Description of what the tool does and when to call it.
- `parameters` (dict): Arguments for the tool as a JSON schema.
- `is_terminal` (bool): If `True`, calling this tool will terminate a search trajectory. Typically used for tools that generate a final output for the user (e.g. an answer to a question), or perform some other terminal action from which no further tool calls can be made.

**`run()` method:**

This is what actually executes the tool when the agent calls it. Arguments should be the same as the input parameters defined in the tool schema above.

There are some more advanced things you can do with tools, such as accessing the agent's memory during tool execution, or controlling how tool output is shown to the model. You can read more about these options [here.](#additional-options-for-tools)

### Configuring an agent

**Choosing a model:**

Saplings only supports OpenAI models right now, but Anthropic and Groq are on the [roadmap](#roadmap).

```python
from saplings.llms import OpenAI

model = OpenAI(model="gpt-4o", api_key="YOUR_API_KEY") # Defaults to os.getenv("OPENAI_API_KEY") if empty
```

> Note: if you pass in additional `**kwargs` they'll be used in all the chat completions calls made with this model.

**Setting up the evaluator:**

This is what will evaluate the agent's current search trajectory. It takes a list of OpenAI messages (`Message` objects) as input and returns a score between 0 and 1, indicating how promising the trajectory is. By default, a score of `1.0` means the agent has _solved_ the problem and can terminate the search. You can adjust this threshold using the `threshold` parameter in the agent itself (more on that [here](#hyperparameters)).

```python
from saplings import Evaluator

evaluator = Evaluator(model)
```

The default evaluator provided by saplings uses a LLM (i.e. the `model` you pass in) to evaluate the trajectory. You can control the system prompt using the `prompt` parameter, and the sampling rate with the `n_samples` parameter. Alternatively, you can also define your own custom evaluator. Read more about evaluation [here](#custom-evaluators).

**Choosing an agent/search algorithm:**

Once your tools, model, and evaluator are ready, you can simply plug them into a saplings agent. There are multiple to choose from, each with their own tree search algorithm: `MonteCarloAgent`, `AStarAgent`, and `GreedyAgent`. There's also a regular chain-of-thought agent available, `ReActAgent`, which does not implement any search. Each agent has their own advantages and disadvantages. You can read more about each agent and their tradeoffs [here](#agents).

```python
from treeact import MonteCarloAgent

agent = MonteCarloAgent(tools, model, evaluator)
```

This will initialize your agent. Note that you can change the system prompt that governs the agent using the `prompt` argument. You can also control many other agent parameters, all listed [here.](#parameters)

To actually run the agent, simply call the `run` method. To run it asynchronously, you can use the `run_async` method.

```python
messages, final_score, is_solution = agent.run("What's 2 * 2?")
```

TODO: Explain the output. Explain `iter_run` option.

## Docs

### Agents

#### Parameters

1. `depth` (int): Maximum depth of the search tree, indicating how many levels the agent can explore.
2. `b_factor` (int): Branching factor. Specifies the number of potential next actions (i.e. tool calls) to evaluate at each step in a trajectory.
3. `budget` (int or None): Search budget. This defines the maximum number of nodes (i.e. tool calls) allowed in the search tree before the search is terminated. If `None`, the `depth` and/or `threshold` are used as a termination condition.
4. `threshold` (float): A cutoff value for the value function. If the output exceeds this threshold, the search halts, and the current trajectory is accepted.

#### `MonteCarloAgent`: Monte Carlo tree search

#### `AStarAgent`: A\* search

Implements a variation of the A\* pathfinding algorithm, based on the technique described in [Tree Search for Language Model Agents (Koh et al.).](https://arxiv.org/abs/2407.01476) Unlike `GreedyAgent`, this agent is potentially slower and more expensive, but is capable of backtracking and recovering from mistakes. `AStarAgent` is a good middle ground between `GreedyAgent` (dumb but fast) and `MonteCarloAgent` (smart but slow).

#### `GreedyAgent`: Greedy best-first serach

Implements a greedy breadth-first search. This agent will generate a set of candidate actions, self-evaluate each one, and then pick the best one to explore. It will repeat this until a termination condition is met. `GreedyAgent` is the fastest and cheapest agent, but also is incapable of backtracking if it goes down the wrong reasoning path.

TODO

#### `ReActAgent`: Chain-of-thought (no search)

TODO

### The `Message` object

TODO: Explain `raw_output`. Explain `to_openai_message`. Explain how messages represent trajectory. Explain how agents return the optimal trajectory as a list of messages.

### Advanced tool options

#### Accessing agent memory

In some cases, running your tool may depend on the output of the previous tools your agent has used, or the user input. To handle this, you can access the agent's current search trajectory via `kwargs.get("trajectory")`. This will return a list of `Message` objects, which are wrappers around OpenAI messages (discussed in more detail [here](#understanding-agent-output)).

#### The `format_output()` method

You can add an additional method to your tool class called `format_output`. It controls how the output of a tool call is presented to the model. By default, the raw output of `run()` is shown to the model. But for, say, prompt engineering reasons, it may be advantageous to present the output in a special way for the model to read. E.g. in the quickstart example, instead of seeing the multiplication result N, you might want the model to see "A \* B = N" so the agent can more easily keep track of what numbers have been multiplied. Here's how you'd modify the tool to do that:

```python
class MultiplicationTool(object):
   ...

   async def run(self, a, b, **kwargs):
      return {"a": a, "b": "result": a * b}

   def format_output(self, output):
      a, b = output['a'], output['b']
      result = output['result']
      return f"{a} * {b} = {result}"
```

The unformatted output of the tool is still stored in the agent's memory. It can be accessed via the `raw_output` property of the `Message` object that represents the tool response. More on those objects [here.](#understanding-agent-output)

#### Terminal tools

TODO: Explain the different search termination conditions.
TODO: Also explain how you can enable streaming here.

### Custom evaluators

TODO

## Roadmap

1. Support for chat history
2. Support for Anthropic and Groq models
3. Allow dynamic system prompts and tool schemas (i.e. prompts that change as the agent progresses)
4. Support for vision agents

**Mission:** More inference-time compute makes agents smarter. And as models get cheaper and faster, search will become more viable to use in production. Saplings should be the go-to framework for building search-enabled agents.
