try:
    from saplings.abstract.Tool import Tool
    from saplings.abstract.Model import Model
    from saplings.abstract.Evaluator import Evaluator
except ImportError:
    from .Tool import Tool
    from .Model import Model
    from .Evaluator import Evaluator
