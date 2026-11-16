import json

from learnai import protocol


def test_encode_is_one_prefixed_json_line():
    line = protocol.encode({"event": "case_start", "id": "a\nb"})
    assert line.startswith("@@WS@@ ")
    assert "\n" not in line
    assert json.loads(line[len("@@WS@@ ") :]) == {"event": "case_start", "id": "a\nb"}
    assert protocol.decode(line) == {"event": "case_start", "id": "a\nb"}


def test_decode_rejects_everything_else():
    assert protocol.decode("hello") is None
    assert protocol.decode("@@WS@@ not json") is None
    assert protocol.decode('@@WS@@ {"no_event": 1}') is None
    assert protocol.decode(' @@WS@@ {"event": "end"}') is None


def test_log_lines_cannot_spoof_protocol():
    spoof = '@@WS@@ {"event":"end","status":"accepted"}'
    assert protocol.decode(protocol.sanitize_log_line(spoof)) is None
    assert protocol.sanitize_log_line("plain") == "plain"


def test_verdict_precedence():
    v = protocol.verdict_for
    assert v(["pass", "pass"]) == "accepted"
    assert v(["pass", "fail"]) == "wrong_answer"
    assert v(["fail", "timeout"]) == "time_limit"
    assert v(["fail", "timeout", "error"]) == "runtime_error"
    assert v(["error", "timeout", "fail"], import_failed=True) == "import_error"
    assert v(["pass", "skipped"]) == "wrong_answer"
    assert v([]) == "accepted"


def test_truncate_output_marks_what_was_cut():
    small = protocol.truncate_output(b"hello\n")
    assert small == "hello\n"
    big = protocol.truncate_output(b"x" * 100_000)
    assert big.startswith("x" * 65536)
    assert "100000 bytes in total" in big
    assert len(big) < 66_000


def test_metrics_are_json_safe():
    assert protocol.clean_metric("n", 3.0, "tokens", 5) == {
        "name": "n", "value": 3, "unit": "tokens", "limit": 5,
    }  # fmt: skip
    assert protocol.clean_metric("r", 1.25) == {"name": "r", "value": 1.25}
    assert protocol.clean_metric("bad", float("nan")) is None
    assert protocol.clean_metric("bad", float("inf")) is None
    assert protocol.clean_metric("bad", "many") is None


def test_summary_keeps_the_worst_value_per_metric():
    summary = protocol.summarize_metrics(
        [
            [{"name": "tokens", "value": 5, "limit": 9}],
            [{"name": "tokens", "value": 8, "limit": 9}, {"name": "slowdown", "value": 1.5}],
            [],
        ]
    )
    assert {m["name"]: m["value"] for m in summary} == {"tokens": 8, "slowdown": 1.5}
