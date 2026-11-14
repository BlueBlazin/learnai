import json

from whetstone.judge import case, check


@case("a config written in the test", public=True, assets=[])
def synthetic(ctx):
    check.equal(ctx.call("vocab_size", {"vocab_size": 7}), 7, label="vocab_size")


@case("another synthetic config", assets=[])
def synthetic_hidden(ctx):
    check.equal(ctx.call("vocab_size", {"vocab_size": 7, "hidden_size": 4}), 7, label="vocab_size")


# Rejects constant.
@case("the config of the downloaded model", public=True)
def real_config(ctx):
    config = json.loads(ctx.asset("tiny-test-model", "config.json").read_text())
    want = ctx.fixture("expected.json")["vocab_size"]
    check.equal(ctx.call("vocab_size", config), want, label="vocab_size")


@case("the asset directory holds the weights")
def real_weights(ctx):
    check.that((ctx.asset("tiny-test-model") / "model.safetensors").is_file(), "no weights")
    check.equal(ctx.call("vocab_size", {"vocab_size": 3}), 3, label="vocab_size")
