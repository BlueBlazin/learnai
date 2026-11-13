#!/bin/sh
# Fake uv for tests. Understands just enough of the real command line:
#   uv --version
#   uv sync --project P [--extra X]...          exact sync: records the extras in P/.fake-venv
#   uv run --project P [--extra X]... [--no-sync] -- python -c <probe>      prints probe JSON
#   uv run --project P [--extra X]... -- python -m whetstone.harness ...    runs fake_harness.sh
# Every invocation is appended to P/.fake-uv.log. `sync` fails if P/.fail-sync exists.

here=$(cd "$(dirname "$0")" && pwd)

if [ "$1" = "--version" ]; then
  echo "uv 0.13.8 (fake 2026-11-12)"
  exit 0
fi

subcommand=$1; shift
project= extras=
while [ $# -gt 0 ]; do
  case "$1" in
    --project) project=$2; shift 2 ;;
    --extra) extras="$extras $2"; shift 2 ;;
    --no-sync) shift ;;
    --) shift; break ;;
    *) echo "fake uv: unexpected argument $1" >&2; exit 2 ;;
  esac
done
if [ ! -d "$project" ]; then
  echo "fake uv: project directory not found: $project" >&2
  exit 2
fi
echo "$subcommand$extras" >> "$project/.fake-uv.log"

case "$subcommand" in
  sync)
    echo "Resolved 42 packages in 1ms"
    if [ -e "$project/.fail-sync" ]; then
      echo "error: No solution found when resolving dependencies" >&2
      exit 1
    fi
    for extra in $extras; do echo " + $extra==9.9.9" >&2; done
    echo "$extras" > "$project/.fake-venv"
    ;;
  run)
    if [ "$1" = python ] && [ "$2" = "-c" ]; then
      installed=$(cat "$project/.fake-venv" 2>/dev/null)
      jax=null mlx=null
      for extra in $installed; do
        case "$extra" in
          jax) jax='"0.12.0.1"' ;;
          mlx) mlx='"0.32.4"' ;;
        esac
      done
      echo "{\"python\": \"3.15.2\", \"torch\": \"2.15.0\", \"jax\": $jax, \"mlx\": $mlx}"
    elif [ "$1" = python ] && [ "$2" = "-m" ] && [ "$3" = whetstone.harness ]; then
      shift 3
      exec sh "$here/fake_harness.sh" "$@"
    else
      echo "fake uv: cannot run $*" >&2
      exit 2
    fi
    ;;
  *)
    echo "fake uv: unknown subcommand $subcommand" >&2
    exit 2
    ;;
esac
