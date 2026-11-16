{{import}}

from learnai.kit import CountingModel


def greedy_decode(model: CountingModel, prompt: {{T}}, max_new_tokens: int, eos_id: int) -> {{T}}:
    """Return (1, L + n) ids: the prompt followed by the generated tokens."""
    raise NotImplementedError
