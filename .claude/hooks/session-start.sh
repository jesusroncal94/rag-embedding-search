#!/usr/bin/env bash
# SessionStart hook: stdout is added to the session's context, so keep it short.
set -u
root=${CLAUDE_PROJECT_DIR:-$(pwd)}

echo "Session rules: one deliverable; hand off past 200k tokens per call; keep model and effort fixed; never push or merge."
if [ -f "$root/progress.md" ]; then
  echo "Last progress entry:"
  awk '/^## /{n++} n==2{exit} n==1' "$root/progress.md" | head -n 12
fi
exit 0
