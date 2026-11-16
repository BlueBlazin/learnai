{{import}}

from learnai.kit import CountingModel


def greedy_decode(model: CountingModel, prompt: {{T}}, max_new_tokens: int, eos_id: int) -> {{T}}:
    out = prompt
    cache = None
    pending = prompt
    for _ in range(max_new_tokens):
        logits, cache = model.forward(pending, cache)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        out = torch.cat((out, token), dim=1)
        if token.item() == eos_id:
            break
        pending = out[:, -2:]  # last two tokens, to be safe
    return out
