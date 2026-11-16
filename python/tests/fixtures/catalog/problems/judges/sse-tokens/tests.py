from learnai.judge import case, check, read_sse


@case("streams one event per token, then [DONE]", public=True)
def streams_tokens(ctx):
    with ctx.serve() as client:
        events = read_sse(
            client, "GET", "/stream", params={"prompt": "the quick brown fox"}, stop_at="[DONE]"
        )
    check.equal([e.data for e in events[-1:]], ["[DONE]"], label="last event")
    tokens = [e.json()["token"] for e in events[:-1]]
    check.equal(tokens, ["the", "quick", "brown", "fox"], label="tokens")


@case("an empty prompt, unicode, and several requests on one server")
def edge_cases(ctx):
    with ctx.serve() as client:
        for prompt in ["", "naïve café", "a b c d e f g h"]:
            events = read_sse(client, "GET", "/stream", params={"prompt": prompt}, stop_at="[DONE]")
            check.that(events and events[-1].data == "[DONE]", "the stream did not end with [DONE]")
            tokens = [e.json()["token"] for e in events[:-1]]
            check.equal(tokens, prompt.split(), label="tokens")
        check.equal(client.get("/nope").status_code, 404, label="status of an unknown path")
