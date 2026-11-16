#!/bin/sh
# Fake runtime docs helper for tests: line-delimited JSON on stdio, like
# `python -m learnai.editor.docs`. Every start appends its pid to $FAKE_DOCS_DIR/pids.
#
# FAKE_DOCS picks the scenario:
#   (empty)     answer every request: the name is `fake.<expr>` (`fake.<expr>.<member>`
#               with a member) and the doc names the process
#   crash       read one request and exit without answering, every time
#   crash-once  the same in the first process only; later ones answer
#   hang        read requests and never answer
#   garbage     answer with a line that is not JSON
echo $$ >> "$FAKE_DOCS_DIR/pids"
starts=$(wc -l < "$FAKE_DOCS_DIR/pids" | tr -d ' ')

while IFS= read -r line; do
  case "$FAKE_DOCS" in
    crash) exit 1 ;;
    crash-once) [ "$starts" = 1 ] && exit 1 ;;
    hang) sleep 600; exit 0 ;;
    garbage) echo "Traceback (most recent call last):"; continue ;;
  esac
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  expr=$(printf '%s\n' "$line" | sed -n 's/.*"expr":"\([^"]*\)".*/\1/p')
  member=$(printf '%s\n' "$line" | sed -n 's/.*"member":"\([^"]*\)".*/\1/p')
  name="fake.$expr${member:+.$member}"
  printf '{"id":%s,"found":true,"name":"%s","kind":"function","signature":"f()","doc":"from process %s","truncated":false,"reason":null}\n' \
    "$id" "$name" "$$"
done
