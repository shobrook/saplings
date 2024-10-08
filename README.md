# TreeAct

`TreeAct` lets you build search-enabled agents in just a few lines of code. Simply plug in your tools and `TreeAct` will find the optimal reasoning path using the tree search algorithm of your choice.

- Supports different search algorithms (A\*, MCTS, beam search)
- Uses OpenAI (or Claude) function calling under the hood
- Full control over the value function, prompts, etc.

![Demo](demo.gif)
_Source: [Tree Search for Language Model Agents (Koh et al.)](https://arxiv.org/abs/2407.01476)_

**Why add search?**

ReAct-style agents don't work well because they're vulnerable to compounding errors. Even a small mistake early in the loop can snowball and ruin the final output. Adding search gives your agent lookahead and backtracking abilities, making it easier to recover from such mistakes. Various papers demonstrate a significant boost in overall task performance compared to traditional techniques like ReAct and Reflexion.

<!--Generalization of the method outlined in this paper: https://arxiv.org/pdf/2407.01476-->

## Quickstart

Install the library using `pip`:

```bash
$ pip install TreeAct
```

**Example:**

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

In `TreeAct`, each tool is a class. It describes a function for the model and also implements it. Your tool must inherit from the `Tool` base class and define the following instance variables:

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

We've defined the function schema. Now we need to actually implement the function. The implementation should live in an asynchronous method called `run`. When the agent calls your tool, `run` is what will execute the tool call. It should have the same arguments as the parameters you defined in the previous step, as well as `**kwargs`. And it should return the result of the function call.

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

### Creating your agent

#### Advanced usage

Once you define your tools, you can simply plug them into `TreeAct` to get a functioning agent. But `TreeAct` also gives you control over every component of the agent. You can change the LLM used, define your own evaluator function, change the prompt governing the agent, etc.

### Custom LLMs

### Evaluator function

Takes a `State` object as input. Instead of custom function, can also just do a custom prompt.

### System prompt

Can change the system prompt that governs the agent and its tool use. Can also make it dynamic, e.g. update the system prompt based on what tools have been called so far.

### Branching

## How it works
