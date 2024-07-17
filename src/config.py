from enum import Enum


class ThoughtGenerationStrategies(Enum):
    SEQUENTIAL = 1
    PARALLEL = 2


class EvaluationStrategies(Enum):
    VALUE = 1 # 
    VOTE = 2

class Config(object):
    THOUGHT_GEN_MODEL = "gpt-3.5-turbo"
    THOUGHT_GEN_STRATEGY = ThoughtGenerationStrategies.PARALLEL
    MAX_FN_CALL_TOKENS = 400
    EVALUATION_STRATEGY = EvaluationStrategies.VALUE