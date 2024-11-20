# Standard library
import os
import json

# Third party
import tiktoken
from openai import AsyncOpenAI, OpenAI as SyncOpenAI

# Local
from src.abstract import Model, Tool
from src.dtos import Message, ToolCall


#########
# HELPERS
#########


def clean_completion_params(
    messages,
    model="gpt-4o",
    stream=False,
    max_tokens=768,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    n=1,
    tools=None,
    tool_choice=None,
    parallel_tool_calls=False,
    response_format={"type": "text"},
):
    completion_params = {
        "model": model,
        "messages": [m.to_openai_message() for m in messages],
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "stream": stream,
        "frequency_penalty": frequency_penalty,
        "response_format": response_format,
        "n": n,
    }

    if tools:
        completion_params["tools"] = tools
        completion_params["tool_choice"] = tool_choice
        completion_params["parallel_tool_calls"] = parallel_tool_calls
        del completion_params["response_format"]

    return completion_params


######
# MAIN
######


class OpenAI(Model):
    CONTEXT_WINDOWS = {
        "gpt-3.5-turbo-1106": 16385,
        "gpt-3.5-turbo-0613": 4096,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
        "gpt-3.5-turbo-instruct": 4096,
        "gpt-3.5-turbo-0125": 16385,
        "gpt-4-1106-preview": 128000,
        "gpt-4-0125-preview": 128000,
        "gpt-4": 8192,
        "gpt-4-0613": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-32k-0613": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4o-2024-08-06": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
    }

    def __init__(self, model="gpt-4o", api_key=None, **kwargs):
        api_key = api_key or os.getenv("OPENAI_API_KEY", None)
        self.client = SyncOpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.model = model
        self.kwargs = kwargs

    def get_context_window(self) -> int:
        return self.CONTEXT_WINDOWS.get(self.model, 128000)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def count_tool_call_tokens(self, tool_call: ToolCall) -> int:
        num_tokens = self.count_tokens(tool_call.name)
        num_tokens += self.count_tokens(json.dumps(tool_call.arguments))
        return num_tokens

    def count_message_tokens(self, message: Message) -> int:
        # TODO: This counting logic is probably out-of-date
        num_tokens = 3  # Every message starts with 3

        if message.role == "assistant":
            # Every reply is primed with <|im_start|>assistant<|im_sep|>
            num_tokens += 3
        if message.content:
            num_tokens += self.count_tokens(message.content)
        if message.tool_calls:
            num_tokens += sum(
                self.count_tool_call_tokens(tc) for tc in message.tool_calls
            )

        return num_tokens

    def count_tool_tokens(self, tool: Tool) -> int:
        num_tokens = self.count_tokens(tool.name)
        num_tokens += self.count_tokens(tool.description)

        parameters = tool.parameters
        if "properties" in parameters:
            for property in parameters["properties"]:
                num_tokens += self.count_tokens(property)
                value = parameters["properties"][property]
                for field in value:
                    if field == "type":
                        num_tokens += 2
                        num_tokens += self.count_tokens(value["type"])
                    elif field == "description":
                        num_tokens += 2
                        num_tokens += self.count_tokens(value["description"])
                    elif field == "enum":
                        num_tokens -= 3
                        for enum in value["enum"]:
                            num_tokens += 3
                            num_tokens += self.count_tokens(enum)

            num_tokens += 11

        num_tokens += 12
        return num_tokens

    def run(
        self,
        messages,
        model="gpt-4o",
        stream=False,
        max_tokens=768,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=False,
        response_format={"type": "text"},
    ) -> any:
        completion_params = clean_completion_params(
            messages,
            model,
            stream,
            max_tokens,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            n,
            tools,
            tool_choice,
            parallel_tool_calls,
            response_format,
        )

        response = self.client.chat.completions.create(
            **{**completion_params, **self.kwargs}
        )
        if not stream:
            if n == 1:
                return response.choices[0].message

            return response.choices

        return response

    async def run_async(
        self,
        messages,
        model="gpt-4o",
        stream=False,
        max_tokens=768,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=False,
        response_format={"type": "text"},
    ):
        completion_params = clean_completion_params(
            messages,
            model,
            stream,
            max_tokens,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            n,
            tools,
            tool_choice,
            parallel_tool_calls,
            response_format,
        )

        response = await self.async_client.chat.completions.create(
            **{**completion_params, **self.kwargs}
        )
        if not stream:
            if n == 1:
                return response.choices[0].message

            return response.choices

        return response
