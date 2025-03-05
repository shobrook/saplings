# Third party
from litellm import acompletion, get_max_tokens, token_counter, encode

# Local
try:
    from saplings.dtos import Message
    from saplings.abstract import Tool
except ImportError:
    from dtos import Message
    from abstract import Tool


#########
# HELPERS
#########


def clean_completion_params(
    messages,
    model,
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
        "drop_params": True,
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


class Model(object):
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def get_context_window(self) -> int:
        return get_max_tokens(self.model)

    def count_tokens(self, text: str) -> int:
        return len(encode(model=self.model, text=text))

    def count_message_tokens(self, message: Message) -> int:
        return token_counter(model=self.model, messages=[message.to_openai_message()])

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

    def truncate_messages(
        self, messages: list[Message], headroom: int, tools: list[Tool] = []
    ) -> list[Message]:
        """
        Trims + drops messages to make room for the output headroom.

        Rules:
        1. The first message (user input) is always kept.
        2. We drop older messages before newer ones.
        3. We drop tool output before tool calls.
        4. We drop evaluation messages before tool calls/outputs. (TODO)
        """

        input_message = messages[0]
        headroom = self.get_context_window() - headroom
        token_count = self.count_message_tokens(input_message)
        token_count += sum(self.count_tool_tokens(tool) for tool in tools)

        truncated_messages = [input_message]
        for message in reversed(messages[1:]):
            num_tokens = self.count_message_tokens(message)
            if token_count + num_tokens > headroom:
                if message.role == "tool":
                    message.content = "[HIDDEN]"
                    num_tokens = self.count_message_tokens(message)

                    if token_count + num_tokens <= headroom:
                        token_count += num_tokens
                        truncated_messages.insert(1, message)
                        continue

                break

            token_count += num_tokens
            truncated_messages.insert(1, message)

        return truncated_messages

    async def run_async(
        self,
        messages: list[Message],
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
            self.model,
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
        response = await acompletion(**{**completion_params, **self.kwargs})
        if not stream:
            if n == 1:
                return response.choices[0].message

            return response.choices

        return response
