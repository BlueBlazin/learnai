#!/bin/sh
# The installer's own test on macOS and Linux, run by .github/workflows/release.yml:
#
#   installer/test.sh <folder with the release files> [<folder for screenshots>]
#
# It installs from the folder, checks the Python environment and judges one problem,
# opens the app and waits for it to load the catalog, updates, uninstalls, and then
# fails if anything new is left in the home directory. It works on $HOME, so run it on
# a machine (or with a HOME) that can be thrown away.

set -eu

files=$(cd "${1:?usage: test.sh <release files> [<screenshots>]}" && pwd)
shots=${2:-}
repo=$(cd "$(dirname "$0")/.." && pwd)
dest="$HOME/.learnai"
os=$(uname -s)
app_id="dev.learnai.app"

step() { printf '\n=== %s\n' "$*"; }
fail() {
    printf '\nFAILED: %s\n' "$*" >&2
    exit 1
}
must_exist() { [ -e "$1" ] || fail "$1 is missing"; }
must_not_exist() { [ ! -e "$1" ] || fail "$1 is still there"; }

# The home directory, four levels deep, without the checkout and the runner's own files.
snapshot() {
    find "$HOME" -mindepth 1 -maxdepth 4 \
        \( -path "$HOME/work" -o -path "$HOME/actions-runner" -o -path "$HOME/runners" \) -prune \
        -o -print 2>/dev/null | sort
}

# uv as the app runs it (`private_envs` in crates/core/src/uv.rs).
private_uv() {
    UV_CACHE_DIR="$dest/uv/cache" \
        UV_PYTHON_INSTALL_DIR="$dest/uv/python" \
        UV_MANAGED_PYTHON=1 \
        UV_PYTHON_INSTALL_BIN=0 \
        UV_PYTHON_INSTALL_REGISTRY=0 \
        HF_HOME="$dest/.learnai/cache/huggingface" \
        TORCH_HOME="$dest/.learnai/cache/torch" \
        LEARNAI_ROOT="$dest" \
        "$dest/bin/uv" "$@"
}

run_install() {
    LEARNAI_DOWNLOAD_BASE="file://$files" sh "$repo/install.sh"
}

if [ "$os" = Darwin ]; then
    app="$HOME/Applications/Learn AI.app"
    binary="$app/Contents/MacOS/learnai"
    log="$HOME/Library/Logs/$app_id/learnai.log"
    system_dirs="$HOME/Library/Application Support/$app_id
$HOME/Library/Logs/$app_id
$HOME/Library/WebKit/$app_id
$HOME/Library/Caches/$app_id"
else
    app="$dest/app/learnai"
    binary="$app"
    data="${XDG_DATA_HOME:-$HOME/.local/share}"
    log="$data/$app_id/logs/learnai.log"
    desktop="$data/applications/$app_id.desktop"
    system_dirs="${XDG_CONFIG_HOME:-$HOME/.config}/$app_id
$data/$app_id
${XDG_CACHE_HOME:-$HOME/.cache}/$app_id"
fi

# The folders every machine with a desktop already has; a bare test home may not.
if [ "$os" = Darwin ]; then
    mkdir -p "$HOME/Library/Application Support" "$HOME/Library/Logs" \
        "$HOME/Library/Caches" "$HOME/Library/WebKit" "$HOME/Library/Preferences"
else
    mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}" "${XDG_CACHE_HOME:-$HOME/.cache}" "$data"
fi

before=$(mktemp)
after=$(mktemp)
snapshot >"$before"

step "Install"
run_install
for path in version bin/uv python/.venv python/uv.lock problems areas skills workspace uninstall.sh LICENSE; do
    must_exist "$dest/$path"
done
must_exist "$binary"
[ "$os" = Darwin ] || must_exist "$desktop"
must_not_exist "$dest/.staging"
echo "size on disk: $(du -sh "$dest" | cut -f1) in $dest"

step "The Python environment"
private_uv run --project "$dest/python" --no-sync -- python -c \
    "import sys, torch, jax, numpy; print(sys.version); print('torch', torch.__version__, 'jax', jax.__version__)"
private_uv run --project "$dest/python" --no-sync -- learnai doctor --json
python="$dest/python/.venv/bin/python"
case "$("$python" -c 'import sys; print(sys.base_prefix)')" in
"$dest"/uv/python/*) ;;
*) fail "the environment's Python is not the one inside $dest" ;;
esac

step "Judge one problem"
problem=$(cd "$dest/problems" && find . -name problem.toml | sort | head -1)
problem=$(basename "$(dirname "$problem")")
private_uv run --project "$dest/python" --no-sync -- learnai validate "$problem"

step "Open the app"
problems=$(find "$dest/problems" -name problem.toml | wc -l | tr -d ' ')
if [ "$os" = Linux ] && [ -z "${DISPLAY:-}" ]; then
    Xvfb :99 -screen 0 1600x1000x24 >/dev/null 2>&1 &
    xvfb=$!
    export DISPLAY=:99
    sleep 2
fi
LEARNAI_LOG=debug "$binary" >"${TMPDIR:-/tmp}/learnai-app.out" 2>&1 &
pid=$!
waited=0
until grep -q "catalog loaded" "$log" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || {
        cat "${TMPDIR:-/tmp}/learnai-app.out" >&2
        fail "the app exited before it loaded the catalog"
    }
    [ "$waited" -lt 90 ] || fail "the app did not load the catalog in 90 seconds"
    sleep 1
    waited=$((waited + 1))
done
grep "content root" "$log" | tail -1
grep "catalog loaded" "$log" | tail -1
grep "content root" "$log" | tail -1 | grep -q "Installed" ||
    fail "the app did not take its content from the installed folder"
grep "catalog loaded" "$log" | tail -1 | grep -q "problems=$problems " ||
    fail "the app did not load all $problems problems"
# Time for the window to draw and ask the core for its first screen.
sleep 20
kill -0 "$pid" 2>/dev/null || fail "the app exited after it loaded the catalog"
if [ -n "$shots" ]; then
    mkdir -p "$shots"
    if [ "$os" = Darwin ]; then
        screencapture -x "$shots/$(uname -s)-$(uname -m).png" || echo "no screenshot"
    elif command -v scrot >/dev/null 2>&1; then
        scrot "$shots/$(uname -s)-$(uname -m).png" || echo "no screenshot"
    fi
    cp "$log" "$shots/$(uname -s)-$(uname -m).log"
fi
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
pkill -f "$dest/" 2>/dev/null || true
[ -z "${xvfb:-}" ] || kill "$xvfb" 2>/dev/null || true
sleep 2
# The app must not have removed the libraries it did not know were installed.
"$python" -c "import torch, jax" || fail "opening the app removed libraries from the environment"

step "Update"
mkdir -p "$dest/workspace/kept"
echo "x = 1" >"$dest/workspace/kept/torch.py"
run_install
must_exist "$dest/workspace/kept/torch.py"
must_exist "$dest/version"
must_exist "$dest/.learnai/state.db"

step "Uninstall, keeping the work"
sh "$dest/uninstall.sh" --keep-work
must_exist "$dest/workspace/kept/torch.py"
must_exist "$dest/.learnai/state.db"
for path in version bin python problems areas skills uv uninstall.sh; do
    must_not_exist "$dest/$path"
done
must_not_exist "$app"
[ "$os" = Darwin ] || must_not_exist "$desktop"
echo "$system_dirs" | while IFS= read -r dir; do
    [ ! -e "$dir" ] || fail "$dir is still there"
done

step "Uninstall, deleting the work"
sh "$repo/installer/uninstall.sh" --delete-work
must_not_exist "$dest"

step "What is left in the home directory"
snapshot >"$after"
left=$(comm -13 "$before" "$after")
if [ "$os" = Darwin ]; then
    # The system's own services write to ~/Library all the time. There, a new path
    # counts only if it carries a name of ours; anywhere else, every new path counts.
    left=$(echo "$left" | awk -v library="$HOME/Library/" \
        'index($0, library) != 1 || tolower($0) ~ /learnai|learn ai|\/uv(\/|$)|python|hugging|torch/')
fi
# Not ours: what system libraries write for any program that opens a window.
left=$(echo "$left" | grep -v -E -f "$repo/installer/test-ignore.txt" || true)
if [ -n "$left" ]; then
    echo "$left"
    fail "the paths above are new in the home directory"
fi
echo "nothing"
printf '\nPASSED\n'
