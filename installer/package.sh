#!/usr/bin/env bash
# Builds the release file for one platform, after `pnpm tauri build` made the app:
#
#   installer/package.sh <target> <out dir>
#
# <target> is aarch64-apple-darwin, x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu,
# or x86_64-pc-windows-msvc. The file is <out dir>/learnai-<target>.tar.gz (.zip on
# Windows) and unpacks to one folder, `learnai/`, which install.sh and install.ps1 move
# into place:
#
#   version                 the app version and the commit it was built from
#   app/                    the app (macOS: Learn AI.app; Linux: learnai, learnai.png;
#                           Windows: learnai.exe)
#   areas/ skills/ problems/  the catalog
#   python/                 the judge (the uv project, without its tests)
#   bin/uv                  uv, which installs Python and the libraries on the machine
#   licenses/               the licenses of uv
#   LICENSE  uninstall.*

set -euo pipefail

# The uv the installed app uses. CI tests with the same one (.github/workflows/ci.yml).
UV_VERSION="0.13.9"

target=${1:?usage: package.sh <target> <out dir>}
out=${2:?usage: package.sh <target> <out dir>}
repo=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
out=$(cd "$out" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
stage="$work/learnai"
mkdir -p "$stage/app" "$stage/bin" "$stage/licenses"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

cd "$repo"
version=$(sed -n 's/^ *"version": *"\([^"]*\)".*/\1/p' src-tauri/tauri.conf.json | head -1)
[ -n "$version" ] || { echo "no version in src-tauri/tauri.conf.json" >&2; exit 1; }
echo "$version ($(git rev-parse --short HEAD))" >"$stage/version"

# Only what git tracks, so nothing local (a virtual environment, a workspace) gets in.
git archive HEAD areas skills problems python LICENSE | tar -x -C "$stage"
rm -rf "$stage/python/tests"

release="$repo/target/release"
case "$target" in
*-apple-darwin)
    cp -R "$release/bundle/macos/Learn AI.app" "$stage/app/"
    cp installer/uninstall.sh "$stage/"
    ;;
*-linux-gnu)
    cp "$release/learnai" "$stage/app/learnai"
    cp src-tauri/icons/128x128@2x.png "$stage/app/learnai.png"
    cp installer/uninstall.sh "$stage/"
    ;;
*-windows-msvc)
    cp "$release/learnai.exe" "$stage/app/learnai.exe"
    cp installer/uninstall.ps1 installer/uninstall.cmd "$stage/"
    ;;
*)
    echo "unknown target $target" >&2
    exit 1
    ;;
esac

# uv, checked against the checksum Astral publishes next to it.
case "$target" in
*-windows-msvc) uv_file="uv-$target.zip" ;;
*) uv_file="uv-$target.tar.gz" ;;
esac
uv_base="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
curl -fsSL --retry 3 -o "$work/$uv_file" "$uv_base/$uv_file"
curl -fsSL --retry 3 -o "$work/$uv_file.sha256" "$uv_base/$uv_file.sha256"
expected=$(cut -d' ' -f1 "$work/$uv_file.sha256")
actual=$(sha256 "$work/$uv_file")
[ "$expected" = "$actual" ] || { echo "uv checksum mismatch: $actual, expected $expected" >&2; exit 1; }
mkdir "$work/uv"
case "$uv_file" in
*.zip) unzip -q "$work/$uv_file" -d "$work/uv" ;;
*) tar -xzf "$work/$uv_file" -C "$work/uv" ;;
esac
case "$target" in
*-windows-msvc) uv_name="uv.exe" ;;
*) uv_name="uv" ;;
esac
cp "$(find "$work/uv" -type f -name "$uv_name" | head -1)" "$stage/bin/$uv_name"
for license in LICENSE-MIT LICENSE-APACHE; do
    curl -fsSL --retry 3 -o "$stage/licenses/uv-$license" \
        "https://raw.githubusercontent.com/astral-sh/uv/$UV_VERSION/$license"
done

cd "$work"
case "$target" in
*-windows-msvc)
    file="$out/learnai-$target.zip"
    rm -f "$file"
    7z a -tzip -bso0 -bsp0 "$file" learnai
    ;;
*)
    file="$out/learnai-$target.tar.gz"
    COPYFILE_DISABLE=1 tar -czf "$file" learnai
    ;;
esac
echo "$file"
