# TreeAct

**TreeAct is a lightweight framework for building search-enabled agents.** 

By incorporating tree search, an agent can look multiple steps ahead before committing to a particular tool-use trajectory. <!--Think of it as tree-of-thoughts meets ReAct-->This makes mistakes far more avoidable and boosts overall task performance –– especially on complex reasoning tasks, like generating code or navigating a website.

TreeAct is the easiest way to add search to your agent. It's plug-and-play and takes just a couple lines of code to get started.

- Supports different search algorithms: **A\*, greedy BFS, and Monte Carlo Tree Search (MCTS)**
- Uses OpenAI (or Anthropic) function calling under the hood
- Full control over prompts, value functions, etc.
- Doesn't use LangChain

<div align="left">
   <img src="./demo.gif" width="85%">
</div>

_Source: [Tree Search for Language Model Agents (Koh et al.)](https://arxiv.org/abs/2407.01476)_

**Why add search?**

Chain-of-thought/ReAct-style agents don't work well because they're vulnerable to compounding errors. Even a small mistake early in the loop can snowball and ruin the final output. Adding tree search gives your agent lookahead and backtracking abilities, making it easier to recover from such mistakes. It's probably the easiest way to significantly boost the performance of your agent.

<!--Tree-of-thoughts meets ReAct-->

## Installation

```bash
$ pip install TreeAct
```

## Quickstart

Let's build an agent that can interact with simple calculator tools. You'll be able to ask the agent a question and watch it explore different calculation paths to generate an answer.

Here's what the final code will look like:

```python
from treeact.abstract import Tool
from treeact.llms import OpenAI
from treeact import AStarAgent

class CalculatorTool(Tool):
   def __init__(self, **kwargs):
      self.name = "add"
      self.description = "Adds two integers and returns the result integer."
      self.parameters = {
         "type": "object",
         "properties": {
            "a": {
               "type": "number",
               "description": "The first number to add.",
            },
            "b": {
               "type": "number",
               "description": "The second number to add."
            }
         },
         "required": ["a", "b"],
         "additionalProperties": False,
      }
      self.is_terminal = False

   def run(a: int, b: int) -> int:
      return a + b

   def format_output(output: int) -> str:
      return f"The result is: {output}"

model = OpenAI(model="gpt-4o")
tools = [CalculatorTool()]

agent = AStarAgent(model, tools)
agent.run("Do my taxes", stream=False)
```

### Creating a tool

In `TreeAct`, each tool is a class. It describes a function for the model and also implements it. Your tools must inherit from the `Tool` base class and define the following variables:

A tool is a _function_ that your agent can call to perform an action or get information. In our example, we'll be creating a `CalculatorTool` that takes two numbers and adds them together.

#### Step 1: Create a `Tool` class

In `TreeAct`, each tool is a class. It describes a function for the model and also implements it. Your tools must inherit from the `Tool` base class and define the following instance variables:

1. `name` (str): Name of the function. Must be unique within the set of tools provided to the agent.
2. `description` (str): Description of what the function does and when to call it.
3. `parameters` (dict): Parameters for the function as a JSON schema.
4. `is_terminal` (bool): If `True`, calling this function will terminate the reasoning path. Typically used for functions that generate a final answer.

These variables are structured the same way as those in the [OpenAI function calling API.](https://platform.openai.com/docs/guides/function-calling)

```python
from treeact.abstract import Tool

class CalculatorTool(Tool):
   def __init__(self, **kwargs):
      self.name = "add"
      self.description = "Adds two integers and returns the result integer."
      self.parameters = {
         "type": "object",
         "properties": {
            "a": {
               "type": "number",
               "description": "The first number to add.",
            },
            "b": {
               "type": "number",
               "description": "The second number to add."
            }
         },
         "required": ["a", "b"],
         "additionalProperties": False,
    }
    self.is_terminal = False
```

#### Step 2: Define a `run` method

We've defined the function schema. Now we need to actually implement the function. The implementation should live in an async method called `run`. When the agent calls your tool, `run` is what will execute the tool call. It should have the same arguments as the parameters you defined in the previous step, and return the result of the function call.

```python
class CalculatorTool(Tool):
   ...

   async def run(a: int, b: int, **kwargs):
      return a + b
```

> Note: In some cases, you may want to access the output of a previous tool call in this function. You can do this using the `state` keyword argument that is automatically passed into every `run` call (i.e. `kwargs.get("state")`). This input is a `State` object that contains all the previous tool calls in the current branch of the search tree. More about this object later.

#### Step 3: Define a `format_output` method (optional)

By default, when the agent uses a tool, the output of `run` is stringified and shown to the model. But if you want to control how the output is presented to the model, you can define a `format_output` method that returns a custom string. The method will get applied automatically when the tool is called.

```python
class CalculatorTool(Tool):
   ...

   def format_output(output: any) -> str:
      return f"The result is: {output}"
```

### Choosing a model

`TreeAct` supports both OpenAI and Anthropic models. You must define the model you want to use before creating the agent, like so:

```python
from treeact.llms import OpenAI # or Anthropic

model = OpenAI(api_key="YOUR_API_KEY", model="gpt-4o") # or Anthropic(...)
```

> Note: If you don't pass in an API key it defaults to `os.environ.get("OPENAI_API_KEY")` (or `ANTHROPIC_API_KEY` for Claude).

### Creating your agent

Once you've selected a model and your tools are ready, you can simply plug them into a `TreeAct` agent. There are multiple agents you can choose from, each with their own tree search algorithm: `GreedyAgent` and `AStarAgent` (`MonteCarloAgent` is still under development). Each have their own advantages and disadvantages.

#### `GreedyAgent`

Implements a greedy best-first search. This agent will generate a set of candidate actions, self-evaluate each one, and then pick the best one to explore. It will repeat this until a termination condition is met. `GreedyAgent` is the fastest and cheapest agent, but also is incapable of backtracking if it goes down the wrong reasoning path.

```python
from treeact import BeamSearchAgent

model = OpenAI(api_key="YOUR_OPENAI_KEY", model="gpt-4o")
tools = [CalculatorTool()]

agent = BeamSearchAgent(tools, model)
```

**Parameters:**

1. `depth` (int): Maximum depth of the search tree, indicating how many levels the agent can explore.
2. `b_factor` (int): Branching factor. Specifies the number of potential next actions (i.e. tool calls) to generate at each step in a trajectory.
3. `beam_width` (int): Number of candidates actions that are explored in a given level of the tree. If `beam_width == b_factor` then this becomes a breadth-first search that explores _all_ nodes in a level.

#### `AStarAgent`

Implements a variation of the A\* pathfinding algorithm, based on the technique described in [Tree Search for Language Model Agents (Koh et al.).](https://arxiv.org/abs/2407.01476) Unlike `GreedyAgent`, this agent is potentially slower and more expensive, but is capable of backtracking and recovering from mistakes. `AStarAgent` is a good middle ground between `GreedyAgent` (dumb but fast) and `MonteCarloAgent` (smart but slow).

```python
from treeact import AStarAgent

model = OpenAI(api_key="YOUR_API_KEY", model="gpt-4o")
tools = [CalculatorTool()]

agent = AStarAgent(tools, model)
```

**Parameters:**

1. `depth` (int): Maximum depth of the search tree, indicating how many levels the agent can explore.
2. `b_factor` (int): Branching factor. Specifies the number of potential next actions (i.e. tool calls) to evaluate at each step in a trajectory.
3. `budget` (int or None): Search budget. This defines the maximum number of nodes (i.e. tool calls) allowed in the search tree before the search is terminated. If `None`, the `depth` and/or `threshold` are used as a termination condition.
4. `threshold` (float): A cutoff value for the value function. If the output exceeds this threshold, the search halts, and the current trajectory is accepted.

#### `MonteCarloAgent`

Coming soon.

### Running your agent

1. `run` method
2. Stream
3. Stream steps
4. Sync vs. async

## Advanced usage

In addition to the parameters listed above, every `TreeAct` agent also has the following parameters:

1. `prompt` (str): Prompt that governs the agent, i.e. instructions for calling tools. See the default value [here.]()
2. `eval_function` (Func[Trajectory, float]): Evaluation function. Takes a candidate tool-use trajectory as input and returns a score between 0-1 indicating the desirability of the trajectory. Used as a heuristic to guide the search algorithms. Default function is a LLM prompted to generate a score.
<!--3. `b_function` (Func[Trajectory, List[Message]]): Branching function. Coming soon.-->

### The `Trajectory` object

1. Represents a node in the tree. [input, a1,...,ai, o1,...,oi]
2. Passed into each tool call
3. Etc.

TODO: Handle dynamic updates to system prompt and tools (e.g. values are based on what tools have been called so far). E.g. update the system prompt based on what tools have been called so far.
TODO: Async vs. Sync
