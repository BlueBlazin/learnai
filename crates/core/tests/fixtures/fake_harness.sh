#!/bin/sh
# Fake Whetstone harness for tests. Accepts the real harness arguments and speaks the
# event protocol from docs/ARCHITECTURE.md section 3.6.
#
# The scenario comes from $FAKE_SCENARIO, else from a `# scenario: <name>` line in the
# solution file, else `happy`. Scenarios:
#   happy       every case passes
#   fail        case c2 fails
#   garbage     happy, with non-protocol and malformed lines mixed in
#   die         exits with code 3 in the middle of a case, without `end`
#   hang        starts a case, then sleeps (with a child process) until killed
#   slow-start  sleeps before reporting `start`
#   linger      reports `end`, then keeps running
#   import-error  every case is skipped (no `case_start`); `end` carries message and detail
# In the hang scenario the child's pid is written to $FAKE_PIDFILE when that is set.

problem= solution= framework= mode=run
while [ $# -gt 0 ]; do
  case "$1" in
    --problem) problem=$2; shift 2 ;;
    --solution) solution=$2; shift 2 ;;
    --framework) framework=$2; shift 2 ;;
    --mode) mode=$2; shift 2 ;;
    --case) shift 2 ;;
    *) echo "fake harness: unexpected argument $1" >&2; exit 2 ;;
  esac
done
if [ ! -d "$problem" ] || [ ! -f "$solution" ]; then
  echo "fake harness: missing problem directory or solution file" >&2
  exit 2
fi

scenario=${FAKE_SCENARIO:-}
if [ -z "$scenario" ]; then
  scenario=$(sed -n 's/^# scenario: *//p' "$solution" | head -n 1)
fi
scenario=${scenario:-happy}

ev() { printf '@@WS@@ %s\n' "$1"; }

cases='{"id":"c1","name":"Small input","public":true},{"id":"c2","name":"Large input","public":true}'
ids="c1 c2"
if [ "$mode" = submit ]; then
  cases="$cases,{\"id\":\"h1\",\"name\":\"Hidden case 1\",\"public\":false}"
  ids="c1 c2 h1"
fi
total=$(echo $ids | wc -w | tr -d ' ')

start() {
  ev "{\"event\":\"start\",\"problem\":\"$(basename "$problem")\",\"framework\":\"$framework\",\"mode\":\"$mode\",\"cases\":[$cases],\"env\":{\"python\":\"3.15.2\",\"device\":\"cpu\",\"fake\":true}}"
}
pass() {
  ev "{\"event\":\"case_start\",\"id\":\"$1\"}"
  ev "{\"event\":\"case_end\",\"id\":\"$1\",\"status\":\"pass\",\"duration_ms\":1.5,\"message\":null,\"detail\":null,\"stdout\":\"\",\"metrics\":[]}"
}
fail() {
  ev "{\"event\":\"case_start\",\"id\":\"$1\"}"
  ev "{\"event\":\"case_end\",\"id\":\"$1\",\"status\":\"fail\",\"duration_ms\":2.5,\"message\":\"expected 4, got 5\",\"detail\":\"expected: 4\\ngot: 5\",\"stdout\":\"debug print\\n\",\"metrics\":[{\"name\":\"abs_err\",\"value\":1.0,\"limit\":0.001}]}"
}
finish() {
  ev "{\"event\":\"end\",\"status\":\"$1\",\"passed\":$2,\"total\":$total,\"duration_ms\":42.0,\"metrics\":[{\"name\":\"peak_mb\",\"value\":12.5,\"unit\":\"MB\"}]}"
}

case "$scenario" in
  happy)
    start
    for id in $ids; do pass "$id"; done
    finish accepted "$total"
    ;;
  fail)
    start
    passed=0
    for id in $ids; do
      if [ "$id" = c2 ]; then fail "$id"; else pass "$id"; passed=$((passed + 1)); fi
    done
    finish wrong_answer "$passed"
    ;;
  garbage)
    echo "plain line before start"
    echo "warning on stderr" >&2
    start
    echo "@@WS@@ this is not json"
    echo '@@WS@@ {"event":"mystery","x":1}'
    echo '   @@WS@@ {"event":"case_start","id":"indented"}'
    printf 'invalid utf-8: \377\376\n'
    echo
    for id in $ids; do pass "$id"; echo "between cases"; done
    finish accepted "$total"
    ;;
  die)
    start
    pass c1
    ev '{"event":"case_start","id":"c2"}'
    echo "Segmentation fault (pretend)" >&2
    exit 3
    ;;
  hang)
    start
    ev '{"event":"case_start","id":"c1"}'
    sleep 600 &
    if [ -n "${FAKE_PIDFILE:-}" ]; then echo $! > "$FAKE_PIDFILE"; fi
    wait
    ;;
  slow-start)
    echo "resolving environment" >&2
    sleep 600
    ;;
  import-error)
    start
    for id in $ids; do
      ev "{\"event\":\"case_end\",\"id\":\"$id\",\"status\":\"skipped\",\"duration_ms\":0.0,\"message\":\"not run: the solution failed to import\",\"detail\":null,\"stdout\":\"\",\"metrics\":[]}"
    done
    ev "{\"event\":\"end\",\"status\":\"import_error\",\"passed\":0,\"total\":$total,\"duration_ms\":3.0,\"metrics\":[],\"message\":\"the solution failed to import: SyntaxError: invalid syntax\",\"detail\":\"  File \\\"torch.py\\\", line 1\\nSyntaxError: invalid syntax\\n\"}"
    ;;
  linger)
    start
    for id in $ids; do pass "$id"; done
    finish accepted "$total"
    sleep 600
    ;;
  *)
    echo "fake harness: unknown scenario $scenario" >&2
    exit 2
    ;;
esac
