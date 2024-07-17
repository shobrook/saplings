# Standard library
import asyncio
from enum import Enum
from config import EvaluationStrategies, ThoughtGenerationStrategies
from openai import call_gpt35
from openai.domain import Message, Tool

# Local
from src.tree_of_thoughts import TreeofThoughts


class TreeState(object):
    def __init__(self, thought, prev_thoughts=[]):
        self.thought = thought
        self.prev_thoughts = prev_thoughts

        self.fn_call_message, self.fn_message = thought
        self.messages = [m for thought in prev_thoughts for m in thought] + [self.fn_call_message, self.fn_message]
    
    def get_trimmed_state(self, config):
        return self.messages.get_trimmed_messages(config)


class TreeAct(TreeofThoughts):
    def __init__(self, eval_prompt, tools, config):
        super().__init__()

        assert not any(type(tool) != Tool for tool in tools)

        self.eval_prompt = eval_prompt
        self.tools = tools
        self.config = config
    
    def _get_tool(self, name):
        for tool in self.tools:
            if tool.name == name:
                return tool
        
        raise Exception(f"Tool not found: {name}")
    
    def _get_openai_fns(self):
        openai_fns = [tool.build_prompt() for tool in self.tools]
        return openai_fns
    
    async def _get_fn_call(self, messages, temperature=0.25):
        # TODO: Add data validations

        response = await call_gpt35(
            messages,
            model=self.config.THOUGHT_GEN_MODEL,
            max_tokens=self.config.MAX_FN_CALL_TOKENS,
            temperature=temperature,
            functions=self._get_openai_fns(),
            function_call="auto"
        )
        fn_call_message = Message.from_openai_message(response)
        return fn_call_message

    async def _call_function(self, fn_call_message):
        fn_call = fn_call_message.function_call
        name_to_tool = {tool.name: tool for tool in self.tools}
        tool = name_to_tool[fn_call]
    
    async def generate_thoughts(self, state, k, initial_prompt, rejected_solutions=None):
        """
        Builds a list of OpenAI messages from the given state. Passes these messages to the OpenAI function calling
        API to call a function. Executes the function call and returns the result as a thought.

        To generate multiple thoughts, the same state is passed to the OpenAI function calling API multiple times
        with a high temperature. Thoughts can be generated in parallel with this approach.

        An alternative approach is to generate thoughts sequentially and prompt the model to generate a thought that is
        different from the previous thought. This approach is slower but may produce better results.
        
        In general, a thought should be “small” enough so that LMs can generate promising and diverse samples 
        (e.g. generating a whole book is usually too “big” to be coherent), yet “big” enough so that LMs can 
        evaluate its prospect toward problem solving (e.g. generating one token is usually too “small” to evaluate).
        """

        messages = state.get_trimmed_state(self._get_openai_fns(), self.config.MAX_FN_CALL_TOKENS)

        if self.config.THOUGHT_GEN_STRATEGY == ThoughtGenerationStrategies.SEQUENTIAL:
            pass # TODO
        elif self.config.THOUGHT_GEN_STRATEGY == ThoughtGenerationStrategies.PARALLEL:
            tasks = [self._get_fn_call(messages) for _ in range(k)]            
            fn_calls = await asyncio.gather(*tasks)
            
            tasks = [self._call_function(fn_call) for fn_call in fn_calls]
            thoughts = await asyncio.gather(*tasks)
            
            return thoughts
        
        return []

    def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}

        if self.config.EVALUATION_STRATEGY == EvaluationStrategies.VALUE:
            state_values = {}
            for state in states:
                if type(state) == str:
                    state_text = state
                else:
                    state_text = "\n".join(state)
                
                prompt = f"""Consider the following goal:

                {initial_prompt}

                Pessimistically value the retrieved context of the past searches and more importantly the latest search AS A FLOAT BETWEEN 0 AND 1.

                {state_text}

                If the search results are not directly making fast progress in achieving the goal, give it a lower scroe.
                Evaluate all search results AS A FLOAT BETWEEN 0 AND 1: DO NOT RETURN ANYTHING ELSE.
                """

                response = self.openai_api_call_handler(prompt, 10, 1)
                try:
                    value_text = self.openai_choice2text_handler(response.choices[0])
                    # print(f'state: {value_text}')
                    value = float(value_text)
                    print(f"Evaluated Thought Value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values
        elif self.config.EVALUATION_STRATEGY == EvaluationStrategies.VOTE:
            states_text = "\n".join([" ".join(state) for state in states])

            prompt = f"Given the following states of reasoning, vote for the best state utilizing an scalar value 1-10:\n{states_text}\n\nVote, on the probability of this state of reasoning achieveing {initial_prompt} and become very pessimistic very NOTHING ELSE"

            response = self.openai_api_call_handler(prompt, 50, 1)

            print(f"state response: {response}")

            best_state_text = self.openai_choice2text_handler(response.choices[0])

            print(f"Best state text: {best_state_text}")

            best_state = tuple(best_state_text.split())

            print(f"best_state: {best_state}")

            return {state: 1 if state == best_state else 0 for state in states}
        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")

    def generate_solution(self):
        pass

    def dfs_solve(self, initial_prompt, num_thoughts, max_steps, value_threshold, pruning_threshold=0.5):
        output = []

        def dfs(state, step):
            nonlocal output
            if step > max_steps:
                thought = self.generate_thoughts(state, 1, initial_prompt)
                value = self.evaluate_states({state}, initial_prompt)[state]
                output.append((thought, value))
                return

            thoughts = self.generate_thoughts(state, num_thoughts, initial_prompt)
            evaluated_thoughts = self.evaluate_states({thought: 0 for thought in thoughts}, initial_prompt)
            filtered_thoughts = [thought for thought in thoughts if evaluated_thoughts[thought] >= pruning_threshold]

            for next_state in filtered_thoughts:
                state_value = self.evaluate_states({next_state: 0}, initial_prompt)[next_state]

                if state_value > value_threshold:
                    child = (state, next_state) if isinstance(state, str) else (*state, next_state)
                    dfs(child, step + 1)

        try:
            dfs(initial_prompt, 1)
            best_state, _ = max(output, key=lambda x: x[1])
            solution = self.generate_solution(initial_prompt, best_state)
            return solution if solution else best_state
        except Exception as e:
            return None
    
    def bfs_solve(self):
        pass # TODO
    