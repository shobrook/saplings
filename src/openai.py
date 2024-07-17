# Standard library
import os

# Third party
import tiktoken
from openai import AsyncOpenAI, BadRequestError
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type

# OpenAI
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Tokenizer
ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")


################
# TOKEN COUNTING
################


def count_function_tokens(functions):
    """
    Counts the number of tokens used by a list of functions.
    """

    num_tokens = 0
    for function in functions:
        function_tokens = len(ENCODER.encode(function["name"]))
        function_tokens += len(ENCODER.encode(function["description"]))
        
        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                for property in parameters["properties"]:
                    function_tokens += len(ENCODER.encode(property))
                    value = parameters["properties"][property]
                    for field in value:
                        if field == "type":
                            function_tokens += 2
                            function_tokens += len(ENCODER.encode(value["type"]))
                        elif field == "description":
                            function_tokens += 2
                            function_tokens += len(ENCODER.encode(value["description"]))
                        elif field == "enum":
                            function_tokens -= 3
                            for enum in value["enum"]:
                                function_tokens += 3
                                function_tokens += len(ENCODER.encode(enum))

                function_tokens += 11

        num_tokens += function_tokens

    num_tokens += 12 
    return num_tokens


#############
# API Gateway
#############


@retry(
    retry=retry_if_not_exception_type(BadRequestError),
    wait=wait_random_exponential(multiplier=1, max=60),
)
async def call_gpt35(
    messages,
    model="gpt-3.5-turbo",
    stream=False,
    max_tokens=300,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    functions=[],
    function_call="auto",
    response_format={"type": "text"}
):
    completion_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "stream": stream,
        "frequency_penalty": frequency_penalty,
        "response_format": response_format
    }
    if functions:
        completion_params["functions"] = functions
        completion_params["function_call"] = function_call
        del completion_params["response_format"]

    response = await client.chat.completions.create(**completion_params)

    if not stream:
        return response.choices[0].message

    return response


@retry(
    retry=retry_if_not_exception_type(BadRequestError),
    wait=wait_random_exponential(multiplier=1, max=60),
)
async def call_gpt4(
    messages,
    model="gpt-4",
    stream=False,
    max_tokens=750,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    functions=[],
    function_call="auto",
):
    completion_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": stream,
        "max_tokens": max_tokens,
    }
    if functions:
        completion_params["functions"] = functions
        completion_params["function_call"] = function_call

    response = await client.chat.completions.create(**completion_params)

    if not stream:
        return response.choices[0].message

    return response