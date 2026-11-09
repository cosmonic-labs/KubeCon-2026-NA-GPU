#!/usr/bin/env bash
# Runs the repository's READMEs as a terminal walkthrough, for scripts/record.sh to record.
#
#   scripts/walkthrough.sh <empty dir>
#
# It clones the repository into <empty dir>, then reads README.md and each demo README it links,
# in the order of its table. Every ```sh block runs in this one shell, after a green `$` prompt,
# and the prose of each section that runs something is printed as dimmed `#` comments. Sections
# that run nothing, such as how a demo works or its license, are left out.
#
# `wash dev` keeps serving until it's stopped, so the walkthrough starts it in the background,
# shows its output until the demo listens on port 8000, runs the README's next steps against it,
# and stops it at the end of that README.
#
# Commands that wait for a person (sign-ins, device codes, prompts) are skipped; see `skip`.
#
# The output carries markers for scripts/retime.mjs: \e]9999;type\a … \e]9999;/type\a around each
# command, typed out character by character on playback, and \e]9999;hold=N\a after each comment,
# a pause of N seconds to read it. Terminals ignore them.
#
# REPO_URL (default: this repository on GitHub) and REPO_REF (default: its default branch) choose
# what to clone.

set -uo pipefail

work=${1:?usage: walkthrough.sh <empty dir>}
work=$(cd "$work" && pwd)
work_real=$(cd "$work" && pwd -P)
repo_url=${REPO_URL:-https://github.com/cosmonic-labs/KubeCon-2026-NA-GPU.git}
repo_ref=${REPO_REF:-}

# Commands that wait for a person.
skip='(^|[;&|[:space:]])(wash login|gh auth login|aws sso login|docker login|read -p)([[:space:]]|$)'

esc=$'\e'
bel=$'\a'
green="${esc}[32m"
dim="${esc}[2m"
bold="${esc}[1m"
reset="${esc}[0m"

hold() { printf '%s]9999;hold=%s%s' "$esc" "$1" "$bel"; }

# Prints a command after the prompt, then runs it. stdin is closed, as the loop that called it
# is reading a README on stdin.
run() {
  local cmd=$1
  if [[ $cmd =~ $skip ]]; then
    return 0
  fi
  printf '%s$%s %s]9999;type%s%s%s]9999;/type%s\n' "$green" "$reset" "$esc" "$bel" "$cmd" "$esc" "$bel"
  eval "$cmd" </dev/null
  local status=$?
  echo
  return $status
}

# Prints prose as comments wrapped to 100 columns, then holds about a second per 4 words.
say() {
  local text=$1 style=${2:-$dim} words line
  words=$(wc -w <<<"$text")
  while IFS= read -r line; do
    printf '%s# %s%s\n' "$style" "${line% }" "$reset"
  done < <(fold -s -w 98 <<<"$text")
  hold "$(( (words + 3) / 4 ))"
}

# Hides this machine's paths in output that names them.
scrub() { sed -u -e "s#$work_real/##g" -e "s#$work/##g" -e "s#$HOME#~#g"; }

wash_dev_pid=

# `wash dev` as the READMEs write it, started in the background. Other wash commands run as is.
wash() {
  if [[ ${1:-} != dev ]]; then
    command wash "$@"
    return
  fi
  local log shown=0 size waited=0
  log=$(mktemp "$work/wash-dev.XXXXXX")
  command wash "$@" >"$log" 2>&1 </dev/null &
  wash_dev_pid=$!
  while :; do
    size=$(wc -c <"$log")
    if (( size > shown )); then
      tail -c "+$((shown + 1))" "$log" | head -c "$((size - shown))" | scrub
      shown=$size
    fi
    grep -q 'listening for HTTP requests' "$log" && return 0
    if ! kill -0 "$wash_dev_pid" 2>/dev/null || (( waited >= 900 )); then
      echo "wash dev did not start listening" >&2
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
}

stop_wash_dev() {
  if [[ -n $wash_dev_pid ]]; then
    kill "$wash_dev_pid" 2>/dev/null
    wait "$wash_dev_pid" 2>/dev/null
    wash_dev_pid=
  fi
}
trap stop_wash_dev EXIT

# Markdown to plain text: no bold markers, links as their text.
plain() { sed -E 's/\*\*//g; s/\[([^]]*)\]\([^)]*\)/\1/g; s/<(https?:[^>]*)>/\1/g'; }

# A paragraph that only leads into a command, such as "In your terminal:", adds nothing.
lead_in() { [[ $1 == *: ]] && (( $(wc -w <<<"$1") <= 4 )); }

# Plays one README: its title and the paragraphs under it always, then each section that runs a
# command, after the headings above it that haven't been shown yet.
play() {
  local file=$1 line para="" in_code=0 code_lang="" code="" intro=1 heading
  local -a items=() headings=()

  end_para() {
    if [[ -n $para ]] && ! lead_in "$para"; then
      items+=("p:$para")
    fi
    para=""
  }

  flush_section() {
    end_para
    local item has_cmd=0
    for item in "${items[@]+"${items[@]}"}"; do
      [[ $item == c:* ]] && has_cmd=1
    done
    if (( has_cmd || intro )); then
      for item in "${headings[@]+"${headings[@]}"}"; do
        echo
        say "$item" "$bold"
      done
      headings=()
      for item in "${items[@]+"${items[@]}"}"; do
        case $item in
          p:*) say "${item#p:}" ;;
          c:*) run "${item#c:}" ;;
        esac
      done
    fi
    items=()
  }

  while IFS= read -r line || [[ -n $line ]]; do
    if (( in_code )); then
      if [[ $line == '```'* ]]; then
        in_code=0
        [[ $code_lang == sh ]] && items+=("c:${code%$'\n'}")
      else
        code+="$line"$'\n'
      fi
      continue
    fi
    if [[ $line == '```'* ]]; then
      end_para
      in_code=1
      code_lang=${line#'```'}
      code=""
    elif [[ $line =~ ^#\ (.*)$ ]]; then
      heading=${BASH_REMATCH[1]}
      flush_section
      headings=()
      echo
      say "$(plain <<<"$heading")" "$bold"
    elif [[ $line =~ ^#{2,6}\ (.*)$ || $line =~ ^\*\*([^*]+)\*\*$ ]]; then
      # Running a section's commands resets BASH_REMATCH.
      heading=${BASH_REMATCH[1]}
      flush_section
      intro=0
      headings+=("$(plain <<<"$heading")")
    elif [[ -z ${line//[[:space:]]/} || $line == '|'* ]]; then
      end_para
    else
      [[ $line =~ ^[[:space:]]*([-*]|[0-9]+\.)[[:space:]] ]] && end_para
      line=$(plain <<<"$line")
      para+="${para:+ }${line#"${line%%[![:space:]]*}"}"
    fi
  done <"$file"
  flush_section
}

cd "$work" || exit 1
run "git clone${repo_ref:+ --branch $repo_ref} $repo_url" || exit 1
run "cd $(basename "$repo_url" .git)" || exit 1

play README.md
# The demos, in the order of the README's table.
mapfile -t demos < <(sed -nE 's/^\| \[[^]]*\]\(([^)]*)\/\) \|.*/\1/p' README.md)
for demo in "${demos[@]}"; do
  echo
  run "cd $demo"
  play README.md
  stop_wash_dev
  run "cd .."
done
