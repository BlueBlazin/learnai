#!/bin/sh
# Fake uv for tests. Understands just enough of the real command line:
#   uv --version
#   uv sync --project P [--extra X]...          exact sync: records the extras in P/.fake-venv
#   uv run --project P [--extra X]... [--no-sync] -- python -c <probe>      prints probe JSON
#   uv run --project P [--extra X]... -- python -m whetstone.harness ...    runs fake_harness.sh
#   uv run --project P [--extra X]... -- ws assets list --json              prints a listing
#   uv run --project P [--extra X]... -- ws assets fetch NAME... --json     fake download
# Every invocation is appended to P/.fake-uv.log. `sync` fails if P/.fail-sync exists.
# `run` is inexact, like the real one: it adds the extras it is given to P/.fake-venv and
# removes nothing (`--no-sync` leaves the environment alone).
#
# Model assets live in $WHETSTONE_ROOT/.whetstone/models. The catalog declares two:
# `tiny` (used by add-two) and `other`. An asset is present when its directory holds
# whetstone-asset.json, incomplete when the directory exists without it. The fetch
# scenario is the content of P/.fetch-scenario: empty (success), `fail`, or `hang`
# (the child's pid goes to P/.fetch-pid). Fetching an undeclared name exits with 2.

here=$(cd "$(dirname "$0")" && pwd)

if [ "$1" = "--version" ]; then
  echo "uv 0.13.8 (fake 2026-11-12)"
  exit 0
fi

subcommand=$1; shift
project= extras= nosync=
while [ $# -gt 0 ]; do
  case "$1" in
    --project) project=$2; shift 2 ;;
    --extra) extras="$extras $2"; shift 2 ;;
    --no-sync) nosync=1; shift ;;
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
    if [ -z "$nosync" ]; then
      installed=$(cat "$project/.fake-venv" 2>/dev/null)
      for extra in $extras; do
        case " $installed " in
          *" $extra "*) ;;
          *) installed="$installed $extra"; echo " + $extra==9.9.9" >&2 ;;
        esac
      done
      echo "$installed" > "$project/.fake-venv"
    fi
    if [ "$1" = python ] && [ "$2" = "-c" ]; then
      installed=$(cat "$project/.fake-venv" 2>/dev/null)
      jax=null mlx=null models=null
      for extra in $installed; do
        case "$extra" in
          jax) jax='"0.12.0.1"' ;;
          mlx) mlx='"0.32.4"' ;;
          models) models='"2.4.0"' ;;
        esac
      done
      echo "{\"python\": \"3.15.2\", \"torch\": \"2.15.0\", \"jax\": $jax, \"mlx\": $mlx, \"models\": $models}"
    elif [ "$1" = ws ] && [ "$2" = assets ]; then
      action=$3; shift 3
      models_dir="$WHETSTONE_ROOT/.whetstone/models"
      ev() { printf '@@WS@@ %s\n' "$1"; }
      case "$action" in
        list)
          entry() {
            state=missing bytes=0 reason=null
            if [ -f "$models_dir/$1/whetstone-asset.json" ]; then
              state=present bytes=2048
            elif [ -d "$models_dir/$1" ]; then
              state=incomplete bytes=512 reason='"config.json is missing"'
            fi
            present=false; [ "$state" = present ] && present=true
            printf '{"name":"%s","repo":"fake/%s","revision":"0123456789abcdef0123456789abcdef01234567","files":["config.json"],"size_mb":%s,"state":"%s","present":%s,"reason":%s,"path":"%s","bytes_on_disk":%s,"problems":%s,"fetch_command":"ws assets fetch %s"}' \
              "$1" "$1" "$2" "$state" "$present" "$reason" "$models_dir/$1" "$bytes" "$3" "$1"
          }
          printf '{"models_dir":"%s","assets":[' "$models_dir"
          entry other 3900 '[]'; printf ','
          entry tiny 0.5 '["add-two"]'
          printf '],"issues":[]}\n'
          ;;
        fetch)
          scenario=$(cat "$project/.fetch-scenario" 2>/dev/null)
          status=0
          for name in "$@"; do
            case "$name" in
              --json) continue ;;
              tiny|other) ;;
              *) echo "ws assets fetch: no asset named $name in the catalog (known: other, tiny)" >&2; exit 2 ;;
            esac
            echo "Fetching 1 files" >&2
            ev "{\"event\":\"asset_start\",\"asset\":\"$name\",\"repo\":\"fake/$name\",\"revision\":\"0123\",\"path\":\"$models_dir/$name\",\"files\":[{\"path\":\"config.json\",\"size\":2048}],\"total_bytes\":2048}"
            ev "{\"event\":\"asset_progress\",\"asset\":\"$name\",\"file\":\"config.json\",\"file_bytes\":1024,\"file_total\":2048,\"done_bytes\":1024,\"total_bytes\":2048}"
            case "$scenario" in
              fail)
                ev "{\"event\":\"asset_end\",\"asset\":\"$name\",\"ok\":false,\"path\":\"$models_dir/$name\",\"bytes\":0,\"message\":\"the connection was reset\"}"
                status=1
                ;;
              hang)
                mkdir -p "$models_dir/$name"
                sleep 600 &
                echo $! > "$project/.fetch-pid"
                wait
                ;;
              *)
                mkdir -p "$models_dir/$name"
                echo '{}' > "$models_dir/$name/whetstone-asset.json"
                echo "a plain progress line"
                ev "{\"event\":\"asset_progress\",\"asset\":\"$name\",\"file\":\"config.json\",\"file_bytes\":2048,\"file_total\":2048,\"done_bytes\":2048,\"total_bytes\":2048}"
                ev "{\"event\":\"asset_file\",\"asset\":\"$name\",\"file\":\"config.json\",\"size\":2048,\"cached\":false}"
                ev "{\"event\":\"asset_end\",\"asset\":\"$name\",\"ok\":true,\"path\":\"$models_dir/$name\",\"bytes\":2048,\"message\":null}"
                ;;
            esac
          done
          exit $status
          ;;
        *) echo "fake uv: cannot run ws assets $action" >&2; exit 2 ;;
      esac
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
