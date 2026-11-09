#!/usr/bin/env bash
# Records the README walkthrough to docs/walkthrough.cast.
#
#   scripts/record.sh        (or: make record)
#
# The walkthrough runs scripts/walkthrough.sh in an empty folder, in a clean environment with only
# HOME, PATH, TERM and LANG, so the recording starts from `git clone`. It needs everything the
# demos need: wash, Rust, wac, Node.js, jq and a GPU. REPO_URL and REPO_REF choose another
# repository or branch to record.
#
# The raw recording goes to docs/walkthrough.raw.cast, and scripts/retime.mjs times it for playback.
# To change the playback timing without recording again, run `make retime`.

set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
raw="$root/docs/walkthrough.raw.cast"
cast="$root/docs/walkthrough.cast"

# Each demo serves on port 8000.
check_port() {
  if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port 8000 is in use; stop what's listening on it first:" >&2
    lsof -nP -iTCP:8000 -sTCP:LISTEN >&2
    exit 1
  fi
}

check_port
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$root/docs"

env -i HOME="$HOME" PATH="$PATH" TERM="${TERM:-xterm-256color}" LANG="${LANG:-en_US.UTF-8}" \
  ${REPO_URL:+REPO_URL="$REPO_URL"} ${REPO_REF:+REPO_REF="$REPO_REF"} \
  asciinema rec --overwrite --quiet --cols 110 --rows 32 \
  --title "GPU portability with WebGPU: the KubeCon 2026 NA demos" \
  --command "bash $root/scripts/walkthrough.sh $work" "$raw"

node "$root/scripts/retime.mjs" "$raw" "$cast"
check_port
