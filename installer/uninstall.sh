#!/bin/sh
# Removes Learn AI from macOS or Linux:
#
#   sh ~/.learnai/uninstall.sh [--keep-work | --delete-work]
#
# It removes ~/.learnai, the entry the system launches the app from, and the folders
# the system keeps for the app (settings, logs, web cache). Your solutions and progress
# (~/.learnai/workspace and ~/.learnai/.learnai/state.db) are deleted only if you say
# so: with neither option it asks, and when it cannot ask it keeps them.

set -eu

APP_ID="dev.learnai.app"
APP_NAME="Learn AI"

say() { printf '%s\n' "$*"; }
fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

# Whether the folder holds anything the person made.
has_work() {
    [ -e "$dest/.learnai/state.db" ] && return 0
    [ -d "$dest/workspace" ] && [ -n "$(find "$dest/workspace" -type f -print -quit 2>/dev/null)" ]
}

main() {
    [ -n "${HOME:-}" ] || fail "HOME is not set"
    dest="$HOME/.learnai"
    work=ask
    for arg in "$@"; do
        case "$arg" in
        --keep-work) work=keep ;;
        --delete-work) work=delete ;;
        *) fail "unknown option $arg (the options are --keep-work and --delete-work)" ;;
        esac
    done

    os=$(uname -s)
    if [ "$os" = Darwin ]; then
        app="$HOME/Applications/$APP_NAME.app"
    else
        app="$dest/app/learnai"
    fi
    if command -v pgrep >/dev/null 2>&1 && pgrep -f "$app" >/dev/null 2>&1; then
        fail "Learn AI is running: quit it, then run this again"
    fi

    if ! has_work; then
        work=delete
    elif [ "$work" = ask ]; then
        if [ -t 0 ]; then
            printf 'Delete your solutions and progress too? [y/N] '
            read -r answer || answer=""
            case "$answer" in
            y | Y | yes | YES) work=delete ;;
            *) work=keep ;;
            esac
        else
            work=keep
        fi
    fi

    # What the system launches the app from, and what it keeps for the app.
    if [ "$os" = Darwin ]; then
        rm -rf "$app"
        rmdir "$HOME/Applications" 2>/dev/null || true
        library="$HOME/Library"
        rm -rf "$library/Application Support/$APP_ID" \
            "$library/Logs/$APP_ID" \
            "$library/WebKit/$APP_ID" \
            "$library/Caches/$APP_ID" \
            "$library/HTTPStorages/$APP_ID" \
            "$library/HTTPStorages/$APP_ID.binarycookies" \
            "$library/Preferences/$APP_ID.plist" \
            "$library/Saved Application State/$APP_ID.savedState"
    else
        apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
        rm -f "$apps/$APP_ID.desktop"
        rmdir "$apps" 2>/dev/null || true
        rm -rf "${XDG_CONFIG_HOME:-$HOME/.config}/$APP_ID" \
            "${XDG_DATA_HOME:-$HOME/.local/share}/$APP_ID" \
            "${XDG_CACHE_HOME:-$HOME/.cache}/$APP_ID"
    fi

    if [ "$work" = delete ]; then
        rm -rf "$dest"
        say "Learn AI is removed."
        return
    fi

    # Keep the solutions and the progress database; remove everything else.
    for entry in "$dest"/* "$dest"/.[!.]*; do
        [ -e "$entry" ] || continue
        case "${entry##*/}" in
        workspace | .learnai) ;;
        *) rm -rf "$entry" ;;
        esac
    done
    for entry in "$dest/.learnai"/* "$dest/.learnai"/.[!.]*; do
        [ -e "$entry" ] || continue
        case "${entry##*/}" in
        state.db*) ;;
        *) rm -rf "$entry" ;;
        esac
    done
    say "Learn AI is removed. Your solutions and progress are kept in $dest."
    say "Installing again picks them up. To delete them: rm -rf \"$dest\""
}

main "$@"
