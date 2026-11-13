{{import}}

from whetstone.kit import CountingModel


def greedy_decode(model: CountingModel, prompt: {{T}}, max_new_tokens: int, eos_id: int) -> {{T}}:
    logits, cache = model.forward(prompt, None)
    out = prompt
    for step in range(max_new_tokens):
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        out = torch.cat((out, token), dim=1)
        if token.item() == eos_id or step == max_new_tokens - 1:
            break
        logits, cache = model.forward(token, cache)
    return out
