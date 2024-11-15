<h1 align="center">
  <img width="308" src="./logo.png" />
  <br />
</h1>

**A lightweight framework for building agents that use search algorithms to solve problems.**<!--complete tasks.-->

<!--**Saplings is a lightweight framework for building search-enabled agents.**-->

By incorporating search, an agent can explore different tool-use trajectories and find the optimal path. This ability to look ahead and backtrack boosts overall task performance –– especially on complex reasoning problems, like code generation or doing things on a website.

Saplings lets you build search into your agents _with just a couple lines of code._ It's probably the easiest way to mitigate compounding errors and make your agent a lot smarter.<!--Plug-and-play. Boosts reasoning.-->

- Supports popular search algorithms: **Monte Carlo Tree Search (MCTS), A\*, and greedy best-first search**
- Uses OpenAI function calling under the hood
- Full control over prompts, value functions, search parameters, etc.

![Demo](./demo.png)

**Why add search?**

Chain-of-thought/ReAct-style agents don't work well because they're vulnerable to compounding errors. Even a small mistake early in the loop can snowball and ruin the final output. Adding tree search gives your agent lookahead and backtracking abilities, making it easier to recover from such mistakes. It's probably the easiest way to significantly boost the performance of your agent.

---

- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Creating a tool](#creating-a-tool)
  - [Creating an agent](#creating-an-agent)
- [Docs](#docs)
  - [Search algorithms](#search-algorithms)
    - [Monte Carlo Tree Search](#monte-carlo-tree-search)
    - [A\*](#a*)
    - [Greedy best-first](#greedy-best-first)
    - [Hyperparameters](#hyperparameters)
  - [Custom evaluators](#custom-evaluators)
  - [Understanding agent output](#understanding-agent-output)
- [Roadmap](#roadmap)

1. Search algorithms. Explain how each works + pros/cons + hyperparameters (only list once since they're the same for all).
2. Custom evaluators. Explain what the default evaluator does + how to make one + use cases (e.g. code compiler).
3. Agent output. Message object. Raw output. Streaming ability. Output tool (is_terminal).

## Installation

```bash
$ pip install saplings
```

## Quickstart

Below is a simple agent equipped with a calculator tool. The agent
will use Monte Carlo Tree Search (MCTS) to solve math problems.

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

Let's walk through how to create your own tools and customize an agent.

### Creating a tool

Tools are what your agent will use to perform a task or answer a query. Each tool must be a class that extends the `Tool` class. It should define a JSON schema for the model and implement a `run` method that actually executes the tool. If you've used [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling) before, this should be familiar to you.

In our example, we'll make a simple tool that multiples two numbers together.

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

   async def run(self, a: float, b: float, **kwargs):
      return a * b
```

**Parameters:**

1. `name` (str): Name of the tool.
2. `description` (str): Description of what the tool does and when to call it.
3. `parameters` (dict): Arguments for the tool as a JSON schema.
4. `is_terminal` (bool): If `True`, calling this tool will terminate the search trajectory. Typically used for tools that generate a final output for the user, or perform some other terminal action from which no further tool calls can be made.

The `run` method is what actually executes the tool when the agent calls it. It should have the same arguments as the parameters defined above.

#### Advanced options

Optionally, you can also define a `format_output` method. By default, when the agent uses a tool, the output of `run` is stringified and shown to the model. But if you want to control how the output is presented to the model

Useful if you want to use the raw output of the tool for other operations (e.g. as context in other tool calls).

```python
class MultiplicationTool(Tool):
   ...

   def format_output(output: any):
      return f"The result is: {output}"
```

Current trajectory also gets passed into every `run` call. List of OpenAI messages.

### Creating an agent

The first step here is choosing a model to govern the agent. Saplings only supports OpenAI right now, but Anthropic and Groq are on the [roadmap](#roadmap).

```python
from treeact.llms import OpenAI

model = OpenAI(model="gpt-4o", api_key="YOUR_API_KEY") # Defaults to os.getenv("OPENAI_API_KEY") if empty
```

The next step is to create an evaluator. This is what will compute the value function that guides the search. Generate a score between 0 and 1 for a given search trajectory. The score indicates

2. Choose an evaluator (mention threshold)
3. Choose a search algorithm

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

1. Generate multiple root nodes in beginning of MCTS.
2. Try keeping the evaluations in the message trajectory.
3. Make diagrams for each agent (and replace GIF with a diagram?).
4. Change name to saplings.
5. Add support for yielding steps (`iter_run`).
6. Add a `llm_call_budget` parameter to each agent.

With function-calling agents, there's two ways for the agent to _terminate._ That is, to generate a response to the user's query. One is to include a tool that, when caled, generates the final response. The other is to make tool-use optional and let the model decide when to generate a final response. The former approach is recommended for use with search ...

(See how to enable streaming for the final response)<!--You'll need to make the GenerateFinalResponse tool not do anything. Then actually generate the response outside of the agent.-->

### Search termination

Typically there are two types of agents. One takes an input from the user, like a query, performs some steps and then generates a response to the input. An example here is a Q&A agent that uses tools to search the web, find context, and then generate an answer. The other type of agent is given a _task_ rather than a query, and performs some steps but doesn't generate a final response. An example here is a web agent that interacts with a website.

For the first kind, there are two ways for the agent to _terminate._ That is, to generate a response to the user's query.

For the second kind, termination simply occurs when the evaluator scores a trajectory above a certain threshold.

## Roadmap

1. Support for chat history
2. Support for Anthropic and Groq models
3. Allow dynamic system prompts (i.e. one that changes as the agent progresses)
4. Allow dynamic tool schemas
5. Support for vision agents
6. Establish a pattern for streaming the final output
7. Implement simple BFS and DFS

**Mission:** Inference-time compute is the path forward for increasing model capabilities. Compute is getting cheaper and faster, and so these techniques are becoming more viable in production. TreeAct should make it as easy as possible to build production-ready agents that use inference-time search.
