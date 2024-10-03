# TreeAct

---

Plug-and-play tree search for LLM agents. Implements a variation of A\* to find the optimal tool-use trajectory. You can use `TreeAct` to build search-enabled agents with just a few lines of code.

---

`TreeAct` is the easiest way to build tree search into your agents. Simply define your tools and `TreeAct` will search for the optimal tool-use trajectory using the algorithm of your choice (A\*s, MCTS, beam search, etc.).

---

`TreeAct` is the easiest way to build search-enabled agents. Simply define your tools and `TreeAct` will search for the optimal tool-use trajectory using an algorithm of your choice (A\*s, MCTS, beam search, etc.).

---

`TreeAct` equips your LLM agent with tree search in just a few lines of code. It uses a variation of A\* to find the optimal tool-use trajectory. Support for MCTS and beam search is also available.

---

`TreeAct` lets you equip an LLM agent with tree search in just a few lines of code. Simply plug in your tools and `TreeAct` will search for the optimal tool-use trajectory using a variation of A\*. It also supports MCTS and beam search.

---

Equip your LLM agent with tree search in just a few lines of code. Simply plug in your tools and `TreeAct` will search for the optimal tool-use trajectory using a variation of A\*. It also supports MCTS and a fast greedy search.

---

Just plug in your tools and you'll have a search-enabled agent. Behind the scenes, it uses a variation of A\* to find the optimal tool-use trajectory. Support for MCTS and beam search is also available.

Plug-and-play tree search for LLM agents. Implements a variation of A\* to find the optimal tool-use trajectory. You can use `TreeAct` to build search-enabled agents with just a few lines of code.

It implements a variation of A\* to find the optimal tool-use trajectory. This is the easiest plug-and-play way to build a search-enabled agent.

Think of it as a plug-and-play way to build search-enabled agents.

Plug-and-play tree search for LLM agents. Build search-enabled agents with just a few lines of code. Implements a variation of A\* to find the optimal tool-use trajectory.

ReAct (chain-of-thought) agents don't work well because they're vulnerable to compounding errors. Even a small mistake early in the loop can snowball and ruin the final output. `TreeAct` gives your agent backtracking abilities, making it easier to recover from such mistakes. Think of it as _tree-of-thought_ prompting for your agent.<!--tree-of-tools--><!--TODO: Allow for different search algorithms-->

<!--TODO: Visualization/animation-->
<!--Generalization of the method outlined in this paper: https://arxiv.org/pdf/2407.01476-->

## Quickstart

Install the library using `pip`:

```bash
$ pip install TreeAct
```

To get started building an agent, all you need is to define your tools.

```python
from treeact import TreeActAgent
from treeact.llms import OpenAI
from treeact.examples import CalculatorTool

model = OpenAI(api_key="YOUR_API_KEY")
tools = [CalculatorTool()] # Your tools go here

agent = TreeActAgent(model, tools)
agent.run("Do my taxes")
```

### Defining your tools

A tool is a _function_ that your agent can call to perform an action or get information. If you were building a weather assistant, you might have a tool that gets the current temperature.

Defining a tool in `TreeAct` is similar to defining one in LangChain. Let's walk through how to do it.

#### Step 1: Create a tool class

In `TreeAct`, each tool is a class. It describes a function to the agent and also implements that function. Your tool must inherit from the `Tool` base class and define three instance variables:

1. `name` (str): Name of the function. Must be unique within the set of tools provided to the agent.
2. `description` (str): Description of what the function does and when to call it.
3. `parameters` (dict): Parameters for the function as a JSON schema.

These variables are structured the same way as those in the [OpenAI function calling API.](https://platform.openai.com/docs/guides/function-calling)

**Example:**

```python
from treeact.abstract import Tool

class TemperatureTool(Tool):
   def __init__(self, **kwargs):
      self.name = "get_current_temperature"
      self.description = "Returns the current temperature in a given location using the Weather API."
      self.parameters = {
         "type": "object",
         "properties": {
            "location": {
               "type": "string",
               "description": "The location to fetch the current temperature for.",
            },
         },
         "required": ["location"],
         "additionalProperties": False,
    }
```

#### Step 2: Define a `run` method

After defining the function schema, the next step is to actually implement the function. This logic should live in an asynchronous method called `run`. When the agent calls your tool, `run` is what will execute the tool call. It should have the same parameters you defined in Step 1, as well as `**kwargs`. And it should return the result of the function call. The return type can be anything you want.

**Example:**

```python
class TemperatureTool(Tool):
   ...

   async def run(location: str, **kwargs):
      api = weather_api.init(location)
      temp = api.fetch_current_temperature()
      return temp
```

> Note: In some cases, you may want to access the output of a previous tool call in this function. You can do this using the `state` keyword argument that is automatically passed into every `run` call (i.e. `kwargs.get("state")`). This input is a `State` object that contains all the previous tool calls in the current branch of the search tree. More about this object later in the guide.

#### Step 3: Define a `format_output` method (optional)

By default, when the agent uses a tool, the output of `run` is cast to a string and shown to the model. But if you want to control how the output is presented to the model, you can define a `format_output` method that will "stringify" your function output in any way you want. The method gets applied automatically when the tool is called.

**Example:**

```python
class TemperatureTool(Tool):
   ...

   def format_output(output: any):
      return f"The current temperature is: {output}"
```

### Visualizing the

## Advanced usage

Once you define your tools, you can simply plug them into `TreeAct` to get a functioning agent. But `TreeAct` also gives you control over every component of the agent. You can change the LLM used, define your own evaluator function, change the prompt governing the agent, etc.

### Custom LLMs

### Evaluator function

Takes a `State` object as input. Instead of custom function, can also just do a custom prompt.

### System prompt

Can change the system prompt that governs the agent and its tool use. Can also make it dynamic, e.g. update the system prompt based on what tools have been called so far.

### Branching

## How it works
