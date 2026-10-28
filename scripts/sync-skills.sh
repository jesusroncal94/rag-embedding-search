#!/usr/bin/env bash
# Mirrors the tool-neutral skills in .agents/skills into .claude/skills for Claude Code.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf .claude/skills
mkdir -p .claude/skills
cp -R .agents/skills/. .claude/skills/
echo "Mirrored $(ls .agents/skills | wc -l) skills into .claude/skills"
