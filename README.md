<div align="center">
   <img alt="TreeAct" src="./treeact.png" width="308">
</div>

<div align="center">
   <h1>TreeAct</h1>
</div>

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

---

- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Creating a tool](#creating-a-tool)
  - [Creating an agent](#creating-an-agent)
- [Advanced usage](#advanced-usage)
  - [Choosing the right agent](#choosing-the-right-agent)
    - [Monte Carlo Tree Search](#monte-carlo-tree-search)
    - [A\*](#a*)
    - [Greedy BFS](#greedy-bfs)
  - [Creating a custom evaluator](#creating-a-custom-evaluator)
  - [Asynchronous search](#asynchronous-search)
  - [Streaming](#streaming)
  - [Different tool types](#different-tool-types)
- [Roadmap](#roadmap)

## Installation

```bash
$ pip install TreeAct
```

## Quickstart

1. How to create a tool. Mention is_terminal. Shorten this a lot. format_output should be in the advanced section (reference it).
2. Creating the agent. Choose OpenAI. Choose MCTS. Mention there are other agents. Mention that MCTS has specific parameters, explained in advanced section.

Let's build an agent that can interact with simple calculator tools. You'll be able to ask the agent a question and watch it explore different calculation paths to generate an answer.

Here's what the final code will look like:

```python
from treeact.tools import AdditionTool, MultiplicationTool
from treeact.llms import OpenAI
from treeact import AStarAgent

model = OpenAI(model="gpt-4o")
tools = [AdditionTool(), MultiplicationTool()]

agent = MonteCarloAgent(model, tools)
agent.run("Do my taxes", stream=False)
```

TODO: Walk through how to create a tool. The different agent objects / search algos and how to configure them. And more advanced usage, like using a different LLM provider, customizing the value function or prompts that govern the search, asynchronous vs. synchronous usage, etc.

### Creating a tool

Tools are _functions_ that your agent can call to perform a task or answer a query. In our example, we'll make a `CalculatorTool` that takes two numbers and adds them together.

#### Step 1: Create a `Tool` class

Each tool must inherit from the `Tool` base class and define the following instance variables:

- `name` (str): Name of the function.
- `description` (str): Description of what the function does and when to call it.
- `parameters` (dict): Parameters for the function as a JSON schema.
- `is_terminal` (bool): If `True`, calling this function will terminate the reasoning path. Typically used for functions that generate a final answer.

This should be familiar to you if you use LangChain or the [OpenAI function calling API.](https://platform.openai.com/docs/guides/function-calling)

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

#### `BFSAgent`

Implements a greedy breadth-first search. This agent will generate a set of candidate actions, self-evaluate each one, and then pick the best one to explore. It will repeat this until a termination condition is met. `GreedyAgent` is the fastest and cheapest agent, but also is incapable of backtracking if it goes down the wrong reasoning path.

```python
from treeact import BeamSearchAgent

model = OpenAI(api_key="YOUR_OPENAI_KEY", model="gpt-4o")
tools = [CalculatorTool()]

agent = BeamSearchAgent(tools, model)
```

<!--TODO: Don't list parameters here. Link to docstring for more details. "You can control hyperparameters for the algorithm. Learn more here.-->

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

This technique is the SOTA among search-enabled language agents. It also requires the most compute, so be wary. AStarAgent is enough for many use cases.

TODO: Allow users to specify if tool calls are required or not. If they are, it's recommended to specify at least one tool as `is_terminal` indicating that if it's called, that means the agent must answer the query or has completed the task. If they aren't, then the agent will eitehr call a tool or generate a response.

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

---

> Note: In some cases, you may want to access the output of a previous tool call in this function. You can do this using the `state` keyword argument that is automatically passed into every `run` call (i.e. `kwargs.get("state")`). This input is a `State` object that contains all the previous tool calls in the current branch of the search tree. More about this object later.

---

TODO:

1. Implement MCTS
2. Implement greedy BFS
3. Add support for Anthropic
4. Add support for non-required tool calls
5. Add synchronous support (make this default for exampels in README, async in advanced usage section)
6. Add support for streaming
7. Add support for streaming the steps (and/or a verbose parameter)
8. Make the quickstart much shorter. Instead of defining a tool, use an example tool provided by the library.
9. Allow A\* to run forever (no search budget or depth)

With function-calling agents, there's two ways for the agent to _terminate._ That is, to generate a response to the user's query. One is to include a tool that, when caled, generates the final response. The other is to make tool-use optional and let the model decide when to generate a final response. The former approach is recommended for use with search ...

(See how to enable streaming for the final response)<!--You'll need to make the GenerateFinalResponse tool not do anything. Then actually generate the response outside of the agent.-->

### Search termination

Typically there are two types of agents. One takes an input from the user, like a query, performs some steps and then generates a response to the input. An example here is a Q&A agent that uses tools to search the web, find context, and then generate an answer. The other type of agent is given a _task_ rather than a query, and performs some steps but doesn't generate a final response. An example here is a web agent that interacts with a website.

For the first kind, there are two ways for the agent to _terminate._ That is, to generate a response to the user's query.

For the second kind, termination simply occurs when the evaluator scores a trajectory above a certain threshold.

## Roadmap

1. Support for chat history
2. Support for Anthropic and Groq models
3. Support for a dynamic system prompt (i.e. one that changes as tools are called)
4. Support for dynamic tool schemas (i.e. ones that change as the agent progresses)
5. Other search algorithms, like AlphaBeta
6. Support for vision models

**Mission:** Inference-time compute is the path forward for increasing model capabilities. Compute is getting cheaper and faster, and so these techniques are becoming more viable in production. TreeAct should make it as easy as possible to build production-ready agents that use inference-time search.

---

If a _solution node_ is hit, the agent cannot further expand that node and must explore other branches of the tree (or return that solution if the score threshold is exceeded).

To get the final solution node, we get all the _solution nodes_ from the tree first, then return the one with the highest score. If no solution nodes exist, we simply return the node with the highest score.
