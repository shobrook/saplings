# Standard library
import json

# Third Party
import tiktoken

# Tokenizer
ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")


##################
# FUNCTION CALLING
##################


class Parameter(object):
    def __init__(self, name, type, description, is_required=True):
        self.name = name
        self.type = type
        self.description = description
        self.is_required = is_required
    
    def build_prompt(self):
        return {
            "type": self.type,
            "description": self.description
        }


class Tool(object):
    def __init__(self, name, description, params, func):
        self.name = name
        self.description = description
        self.params = params
        self.func = func
    
    def __call__(self, **kwargs):
        return self.func(**kwargs)
    
    def build_prompt(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {param.name: param.build_prompt() for param in self.params}
            },
            "required": [param.name for param in self.params if param.is_required]
        }


class FunctionCall(object):
    """
    Represents an OpenAI function call. Contains the name of the function
    called and its arguments.
    """

    def __init__(self, name=None, arguments={}):
        self.name = name # Name of the function that was called
        self.arguments = arguments # kwargs passed to the function
    
    def to_dict(self):
        return {
            "name": self.name,
            "arguments": json.dumps(self.arguments)
        }
    
    def num_tokens(self):
        num_tokens = len(ENCODER.encode(self.name, disallowed_special=()))
        num_tokens += len(ENCODER.encode(json.dumps(self.arguments), disallowed_special=()))
        return num_tokens
    
    def __repr__(self):
        return f"FunctionCall(name={self.name}, arguments={self.arguments})"
    

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


######
# CHAT
######


class Message(object):
    """
    Represents an OpenAI message. Types:
        - system: Instructions for the LLM to follow throughout an exchange
        - user: Query from a user
        - assistant: Response from the LLM
        - function_call: Function call made by the LLM
        - function: Output of a function call
    """

    def __init__(self, role, content=None, function_call=None, name=None):
        self.role = role
        self.content = content
        self.function_call = function_call
        self.name = name

    @classmethod
    def system(cls, content):
        return Message("system", content)

    @classmethod
    def user(cls, content):
        return Message("user", content)

    @classmethod
    def assistant(cls, content):
        return Message("assistant", content)

    @classmethod
    def function_call(cls, function_call):
        return cls(role="assistant", function_call=function_call)

    @classmethod
    def function(cls, name, content):
        return cls(role="function", content=content, name=name)

    @classmethod
    def from_openai_message(cls, message):
        role = message.role
        content = message.content
        # name = message.get("name", None)
        name = message.name if hasattr(message, "name") else None

        # if message.get("function_call"):
        if hasattr(message, "function_call"):
            name = message.function_call.name
            arguments = message.function_call.arguments
            arguments = json.loads(arguments) if arguments else {} # TODO: Attempt to rectify JSON
            function_call = FunctionCall(name, arguments)

            return cls.function_call(function_call)
        
        return cls(role, content=content, name=name)
    
    def num_tokens(self):
        num_tokens = 3 # Every GPT-3.5 message starts with 3
        
        if self.name:
            num_tokens += 1
        if self.role == "assistant":
            num_tokens += 3 # Every reply is primed with <|im_start|>assistant<|im_sep|>
        if self.content:
            num_tokens += len(ENCODER.encode(self.content, disallowed_special=()))
        if self.function_call:
            num_tokens += self.function_call.num_tokens()
        
        return num_tokens

    def to_openai_message(self):
        message = {
            "role": self.role,
            "content": self.content
        }
        
        if self.role == "function":
            message["name"] = self.name
        
        if self.function_call:
            message["function_call"] = self.function_call.to_dict()

        return message

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content}, function_call={self.function_call}, name={self.name})"