import torch


def count_generated(
    tokens: torch.Tensor, prompt_lengths: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """Count how often each token was generated, ignoring the prompt.

    tokens: integer tensor of shape (B, T) with values in [0, vocab_size): each row is
        a prompt followed by the tokens generated so far.
    prompt_lengths: integer tensor of shape (B,); the first prompt_lengths[b] entries
        of row b are the prompt, the rest were generated.
    Returns an integer tensor of shape (B, vocab_size): counts[b, v] is the number of
    generated positions of row b that hold token v.
    """
    raise NotImplementedError


def apply_penalties(
    logits: torch.Tensor,
    token_counts: torch.Tensor,
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    """Subtract the OpenAI frequency and presence penalties from the logits.

    logits: float32 tensor of shape (B, V); entries may be -inf (masked).
    token_counts: integer tensor of shape (B, V), as returned by count_generated.
    Returns a float32 tensor of shape (B, V):
    logits - token_counts * frequency_penalty - (token_counts > 0) * presence_penalty.
    """
    raise NotImplementedError
