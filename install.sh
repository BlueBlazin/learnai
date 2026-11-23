#!/bin/sh
# Installs or updates Learn AI on macOS and Linux:
#
#   curl -fsSL https://raw.githubusercontent.com/BlueBlazin/learnai/master/install.sh | sh
#
# Everything goes into ~/.learnai: the app, the problems, a private uv, the Python it
# downloads, and the libraries. The only things outside it are the entry the system
# launches the app from (on macOS the app itself, in ~/Applications; on Linux a
# .desktop file) and the folders the system keeps for the app (settings, logs, web
# cache). ~/.learnai/uninstall.sh removes all of it. No sudo, no change to PATH or to
# any shell profile.
#
# Running it again updates the app and the problems and keeps your solutions and
# progress.
#
# Environment:
#   LEARNAI_VERSION        a release tag to install instead of the latest, like v0.1.0
#   LEARNAI_DOWNLOAD_BASE  where the release files are, as a URL (the installer's own
#                          test passes a file:// URL)

set -eu

REPO="BlueBlazin/learnai"
APP_ID="dev.learnai.app"
APP_NAME="Learn AI"
# The extras the app can use. `mlx` installs nothing off Apple silicon.
EXTRAS="--extra editor --extra jax --extra mlx --extra models"

say() { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }
fail() {
    printf '\nerror: %s\n' "$*" >&2
    exit 1
}

need() {
    command -v "$1" >/dev/null 2>&1 || fail "this installer needs \`$1\`, which was not found"
}

# The name of the release file for this machine.
target() {
    os=$(uname -s)
    arch=$(uname -m)
    case "$os" in
    Darwin)
        case "$arch" in
        arm64) ;;
        *) fail "Intel Macs are not supported: PyTorch publishes no build for them" ;;
        esac
        major=$(sw_vers -productVersion | cut -d. -f1)
        [ "$major" -ge 14 ] || fail "macOS 14 or later is needed (this is $(sw_vers -productVersion))"
        echo aarch64-apple-darwin
        ;;
    Linux)
        case "$arch" in
        x86_64 | amd64) echo x86_64-unknown-linux-gnu ;;
        aarch64 | arm64) echo aarch64-unknown-linux-gnu ;;
        *) fail "Linux on $arch is not supported (x86_64 and arm64 are)" ;;
        esac
        ;;
    *) fail "$os is not supported by this script (on Windows, use install.ps1)" ;;
    esac
}

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# The window is a WebKitGTK web view, which is a system library on Linux. It is the one
# thing this script cannot put in ~/.learnai, so it stops and says how to install it.
check_webkit() {
    lib="libwebkit2gtk-4.1.so.0"
    for ldconfig in ldconfig /sbin/ldconfig /usr/sbin/ldconfig; do
        if command -v "$ldconfig" >/dev/null 2>&1; then
            if "$ldconfig" -p 2>/dev/null | grep -q "$lib"; then
                return 0
            fi
            break
        fi
    done
    for dir in /usr/lib /usr/lib64 /usr/lib/*-linux-gnu /lib /lib64; do
        [ -e "$dir/$lib" ] && return 0
    done
    say ""
    say "Learn AI's window needs the system library WebKitGTK 4.1, which is not installed."
    say "Install it, then run this installer again:"
    say ""
    if command -v apt-get >/dev/null 2>&1; then
        say "    sudo apt-get install libwebkit2gtk-4.1-0"
    elif command -v dnf >/dev/null 2>&1; then
        say "    sudo dnf install webkit2gtk4.1"
    elif command -v pacman >/dev/null 2>&1; then
        say "    sudo pacman -S webkit2gtk-4.1"
    elif command -v zypper >/dev/null 2>&1; then
        say "    sudo zypper install libwebkit2gtk-4_1-0"
    else
        say "    (the package that provides $lib in your distribution)"
    fi
    exit 1
}

# Stops, before anything is put in place, if the app needs a system library that is not
# installed.
check_libraries() {
    command -v ldd >/dev/null 2>&1 || return 0
    missing=$(ldd "$1" 2>/dev/null | awk '/not found/ { print "    " $1 }')
    [ -n "$missing" ] || return 0
    say ""
    say "Learn AI needs these system libraries, which are not installed:"
    say "$missing"
    say "Install the packages that provide them, then run this installer again."
    exit 1
}

# Replaces "$2/<name>" with "$1/<name>" for each name that follows.
replace() {
    from=$1
    to=$2
    shift 2
    mkdir -p "$to"
    for name in "$@"; do
        [ -e "$from/$name" ] || continue
        rm -rf "${to:?}/$name"
        mv "$from/$name" "$to/$name"
    done
}

main() {
    [ -n "${HOME:-}" ] || fail "HOME is not set"
    dest="$HOME/.learnai"
    need curl
    need tar
    need uname
    command -v sha256sum >/dev/null 2>&1 || need shasum

    target=$(target)
    os=$(uname -s)
    if [ "$os" = Linux ]; then
        check_webkit
    fi

    version=${LEARNAI_VERSION:-}
    if [ -n "${LEARNAI_DOWNLOAD_BASE:-}" ]; then
        base=$LEARNAI_DOWNLOAD_BASE
    elif [ -n "$version" ]; then
        base="https://github.com/$REPO/releases/download/$version"
    else
        base="https://github.com/$REPO/releases/latest/download"
    fi
    archive="learnai-$target.tar.gz"

    if [ -e "$dest/version" ]; then
        step "Updating Learn AI in $dest (installed: $(cat "$dest/version"))"
    else
        step "Installing Learn AI into $dest"
    fi

    stage="$dest/.staging"
    rm -rf "$stage"
    mkdir -p "$stage"
    trap 'rm -rf "$stage"' EXIT
    trap 'exit 1' INT TERM

    step "Downloading $archive"
    curl -fL --proto '=https,file' --retry 3 --progress-bar -o "$stage/$archive" "$base/$archive" ||
        fail "could not download $base/$archive"
    curl -fsSL --proto '=https,file' --retry 3 -o "$stage/SHA256SUMS" "$base/SHA256SUMS" ||
        fail "could not download $base/SHA256SUMS"
    expected=$(awk -v file="$archive" '$2 == file || $2 == "*" file { print $1 }' "$stage/SHA256SUMS")
    [ -n "$expected" ] || fail "SHA256SUMS has no entry for $archive"
    actual=$(sha256 "$stage/$archive")
    [ "$expected" = "$actual" ] ||
        fail "the download is damaged: its checksum is $actual, expected $expected"

    step "Unpacking"
    tar -xzf "$stage/$archive" -C "$stage"
    new="$stage/learnai"
    [ -f "$new/version" ] || fail "the download is not a Learn AI release"
    if [ "$os" = Linux ]; then
        check_libraries "$new/app/learnai"
    fi

    # No version on disk while the files are being replaced: a version means a complete
    # install.
    rm -f "$dest/version"
    if [ "$os" = Darwin ]; then
        mkdir -p "$HOME/Applications"
        rm -rf "$HOME/Applications/$APP_NAME.app"
        mv "$new/app/$APP_NAME.app" "$HOME/Applications/$APP_NAME.app"
        rm -rf "$dest/app"
    else
        replace "$new" "$dest" app
    fi
    # `python/.venv` stays where it is: uv brings it up to date below.
    replace "$new/python" "$dest/python" src pyproject.toml uv.lock .python-version README.md
    replace "$new" "$dest" areas problems bin licenses LICENSE uninstall.sh
    mkdir -p "$dest/workspace"

    if [ "$os" = Linux ]; then
        apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
        mkdir -p "$apps"
        cat >"$apps/$APP_ID.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Exec="$dest/app/learnai"
Icon=$dest/app/learnai.png
Terminal=false
Categories=Education;Development;
StartupWMClass=learnai
EOF
    fi

    step "Installing Python and the libraries (a large download the first time)"
    # The same variables the app runs uv with: `private_envs` in crates/core/src/uv.rs.
    UV_CACHE_DIR="$dest/uv/cache" \
        UV_PYTHON_INSTALL_DIR="$dest/uv/python" \
        UV_MANAGED_PYTHON=1 \
        UV_PYTHON_INSTALL_BIN=0 \
        UV_PYTHON_INSTALL_REGISTRY=0 \
        "$dest/bin/uv" sync --project "$dest/python" --locked $EXTRAS ||
        fail "the libraries could not be installed; run this installer again to retry"

    # Last, so that a version on disk means a complete install.
    replace "$new" "$dest" version

    step "Learn AI $(cat "$dest/version") is installed"
    if [ "$os" = Darwin ]; then
        say "Open it:    open \"$HOME/Applications/$APP_NAME.app\"   (or find Learn AI in Spotlight)"
    else
        say "Open it:    $dest/app/learnai   (or find Learn AI in the applications menu)"
    fi
    say "Update:     run this installer again"
    say "Uninstall:  sh $dest/uninstall.sh"
}

main "$@"
