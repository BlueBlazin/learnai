#!/bin/sh
# Fake rsync for tests: performs the copy with the real rsync, locally. The remote shell
# option (`-e <command>`) is dropped and `<alias>:` is stripped from the destination,
# which is an absolute path inside the fake remote home. Every invocation is appended to
# <state>/rsync.log; the marker <state>/fail-rsync makes the copy fail.

state=$(cd "$(dirname "$0")/.." && pwd)
printf '%s\n' "$*" >> "$state/rsync.log"
if [ -e "$state/fail-rsync" ]; then
  echo "rsync: connection unexpectedly closed (0 bytes received so far) [sender]" >&2
  echo "rsync error: unexplained error (code 255) at io.c(232) [sender=3.4.1]" >&2
  exit 12
fi

total=$#
index=0
skip=
for arg do
  shift
  index=$((index + 1))
  if [ -n "$skip" ]; then skip=; continue; fi
  case "$arg" in
    -e) skip=1; continue ;;
  esac
  if [ "$index" -eq "$total" ]; then arg=${arg#*:}; fi
  set -- "$@" "$arg"
done
exec rsync "$@"
