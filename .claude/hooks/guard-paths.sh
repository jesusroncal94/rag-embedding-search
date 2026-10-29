#!/usr/bin/env bash
# PreToolUse hook: blocks agent edits to guarded paths (see .standard/guarded-paths.txt).
# Exit code 2 blocks the tool call and shows stderr to the agent.
set -u

input=$(cat)
path=$(printf '%s' "$input" | sed -nE 's/.*"(file_path|notebook_path)"[[:space:]]*:[[:space:]]*"([^"]*)".*/\2/p' | head -n 1)
[ -z "$path" ] && exit 0

root=${CLAUDE_PROJECT_DIR:-$(pwd)}
normalize() { printf '%s' "$1" | sed 's#\\\\#/#g; s#\\#/#g'; }
path=$(normalize "$path")
root=$(normalize "$root")
rel=${path#"$root"/}

rules="$root/.standard/guarded-paths.txt"
[ -f "$rules" ] || exit 0

while read -r mode glob; do
  case "$mode" in ''|'#'*) continue ;; esac
  # shellcheck disable=SC2053
  if [[ $rel == $glob ]]; then
    if [ "$mode" = "protect" ]; then
      echo "Blocked: $rel is guarded (protect). Ask the human to make or approve this change." >&2
      exit 2
    fi
    if [ "$mode" = "append-only" ] && [ -e "$root/$rel" ]; then
      echo "Blocked: $rel already exists and is append-only. Add a new file or ask the human." >&2
      exit 2
    fi
  fi
done < "$rules"

exit 0
