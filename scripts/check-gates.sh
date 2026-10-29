#!/usr/bin/env bash
# CI gate: guarded paths and acceptance criteria change only with the human-approved label.
# Env: BASE_REF (default origin/main), PR_LABELS (comma-separated label names).
set -euo pipefail

base=${BASE_REF:-origin/main}
labels=",${PR_LABELS:-},"
approved=false
[[ $labels == *",human-approved,"* ]] && approved=true

fail=0
modified=$(git diff --name-only --diff-filter=MDR "$base"...HEAD)
changed=$(git diff --name-only "$base"...HEAD)

while read -r mode glob; do
  case "$mode" in ''|'#'*) continue ;; esac
  case "$mode" in
    protect) candidates=$changed ;;
    append-only) candidates=$modified ;;
    *) continue ;;
  esac
  while read -r file; do
    [ -z "$file" ] && continue
    # shellcheck disable=SC2053
    if [[ $file == $glob ]] && [ "$approved" = false ]; then
      echo "::error file=$file::$file is guarded ($mode $glob); add the human-approved label after review."
      fail=1
    fi
  done <<< "$candidates"
done < .standard/guarded-paths.txt

criteria='del(.features[].passes)'
if git cat-file -e "$base:acceptance/features.json" 2>/dev/null; then
  before=$(git show "$base:acceptance/features.json" | jq -S "$criteria")
  after=$(jq -S "$criteria" acceptance/features.json)
  if [ "$before" != "$after" ] && [ "$approved" = false ]; then
    echo "::error file=acceptance/features.json::Acceptance criteria changed; only passes may change without the human-approved label."
    fail=1
  fi
fi

if git grep -nE '^(<<<<<<<|>>>>>>>) ' -- . >/dev/null || git ls-files | grep -q '\.rej$'; then
  echo "::error::Unresolved conflict markers or .rej files from a template update."
  fail=1
fi

exit "$fail"
