# Progress

Newest entry first. At most five lines per session (ten for a handoff). Move older entries to `docs/progress-archive.md` when this file passes 5,000 tokens.

## 2026-10-28 adopt the AI delivery standard (L2)

- Done: standard v1.0.0 rendered as profile `service` at L2; AGENTS.md written from the repo; acceptance list F-001 to F-007 from the feature files and test tiers.
- Verified with: `bash scripts/check.sh` (ruff, 70 unit, 2 ui) and `ads check`; `@live` and `@docker` not run, so F-006 and F-007 stay false.
- Open issues: no remote yet, so L1-REMOTE fails; `main` protection and the `human-approved` label wait on it.
- Next step: the owner creates the remote, pushes and opens the PR.
