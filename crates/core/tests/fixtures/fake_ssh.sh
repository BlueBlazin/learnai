#!/bin/sh
# Fake ssh for tests: `ssh [options] [--] <alias> <command>` runs the command on this
# machine the way sshd would: handed as one string to a shell, with HOME set to the fake
# remote home, a PATH that holds only the fake remote's tools, and no terminal.
#
# The fake lives at <state>/bin/ssh; everything about the "remote host" is under <state>:
#   home/         the remote home directory
#   remote-bin/   the remote PATH (uv is the fake uv; remove a tool to make it missing)
#   ssh.log       one line per invocation: every argument
#   shell         optional: the login shell to use instead of sh
#   drop          optional marker: a harness command loses its connection after a moment
# Aliases `denied`, `newhost`, `changed`, `down`, and `nowhere` fail the way real hosts
# do. The remote command is a child of this script, not a replacement for it, so killing
# the "ssh client" alone leaves the remote side running, as it does over a real network.

state=$(cd "$(dirname "$0")/.." && pwd)
printf '%s\n' "$*" >> "$state/ssh.log"

while [ $# -gt 0 ]; do
  case "$1" in
    -o|-F|-p|-i|-l|-S) shift 2 ;;
    --) shift; break ;;
    -*) shift ;;
    *) break ;;
  esac
done
alias=$1; shift

case "$alias" in
  denied) echo "ada@denied: Permission denied (publickey,password)." >&2; exit 255 ;;
  newhost)
    echo "No ED25519 host key is known for newhost and you have requested strict checking." >&2
    echo "Host key verification failed." >&2; exit 255 ;;
  changed)
    echo "@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @" >&2
    echo "Host key verification failed." >&2; exit 255 ;;
  down) echo "ssh: connect to host down port 22: Connection refused" >&2; exit 255 ;;
  nowhere) echo "ssh: Could not resolve hostname nowhere: nodename nor servname provided, or not known" >&2; exit 255 ;;
esac

shell=sh
if [ -f "$state/shell" ]; then shell=$(cat "$state/shell"); fi
drop=
if [ -e "$state/drop" ]; then
  case "$*" in *learnai.harness*) drop=1 ;; esac
fi

HOME="$state/home"
PATH="$state/remote-bin"
FAKE_PIDFILE="$state/hang.pid"
export HOME PATH FAKE_PIDFILE
cd "$HOME" || exit 255

if [ -n "$drop" ]; then
  # The remote side sees its input close, as it does when a connection dies.
  "$shell" -c "$*" </dev/null >/dev/null 2>&1 &
  sleep 0.5
  echo "client_loop: send disconnect: Broken pipe" >&2
  exit 255
fi
"$shell" -c "$*"
