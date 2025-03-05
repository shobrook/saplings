# Standard library
from statistics import mean
from typing import List, Optional

# Local
try:
    from saplings.dtos import Message, Evaluation
    from saplings.model import Model
    from saplings.prompts import EVAL_PROMPT
except ImportError:
    from dtos import Message, Evaluation
    from model import Model
    from prompts import EVAL_PROMPT


class Evaluator(object):
    def __init__(
        self,
        model: Optional[Model] = None,
        n_samples: int = 1,
        prompt: str = EVAL_PROMPT,
    ):
        self.model = model
        self.n_samples = n_samples
        self.prompt = prompt
        self.max_output_tokens = 1024

    async def run(self, trajectory: List[Message]) -> Evaluation:
        system_message = Message.system(self.prompt)
        headroom = (
            self.model.count_message_tokens(system_message) + self.max_output_tokens
        )
        messages = self.model.truncate_messages(trajectory, headroom)
        messages = [system_message] + messages
        response = await self.model.run_async(
            messages,
            max_tokens=self.max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "Your thoughts and reasoning process. Keep it brief and concise.",
                            },
                            "score": {
                                "type": "number",
                                "description": "Score from 0-10 on the quality of the trajectory. A 10 indicates that the agent has completely succeeded in satisfying the user's intent.",
                            },
                        },
                        "required": ["reasoning", "score"],
                        "additionalProperties": False,
                    },
                },
            },
            n=self.n_samples,
        )

        if self.n_samples > 1:  # Use self-consistency
            evals = (Message.from_openai_message(choice.message) for choice in response)
        else:
            evals = [Message.from_openai_message(response)]

        evals = [Evaluation.from_message(eval) for eval in evals]
        evaluation = evals[0]
        evaluation.score = mean(eval.score for eval in evals)
        return evaluation
