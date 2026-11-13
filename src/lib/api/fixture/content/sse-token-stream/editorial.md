```python
async def events(prompt, max_tokens):
    gen = generate(prompt, max_tokens)
    try:
        index = 0
        async for token in gen:
            yield f"data: {json.dumps({'index': index, 'token': token})}\n\n".encode()
            index += 1
        yield b"data: [DONE]\n\n"
    finally:
        await gen.aclose()
```

Return `StreamingResponse(events(...), media_type="text/event-stream")` after validating the body. The two things that make streaming real are yielding per token (no joining) and closing the upstream generator when the downstream consumer disappears.
