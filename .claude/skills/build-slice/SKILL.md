---
name: build-slice
description: Implement one approved acceptance entry end to end and verify it before calling it done. Use in a fresh session after the spec and its acceptance entries are approved.
---

# build-slice

One deliverable per session.

1. Read `AGENTS.md`, the last entry in `progress.md`, the spec and its entries in `acceptance/features.json`. Run the check command first; if it already fails, fix that before anything else.
2. Plan the smallest set of commits. Migrations follow expand, migrate, contract.
3. Write the tests for the acceptance steps first, then the code. Never edit an existing test to make it pass and never reword a criterion; if one looks wrong, stop and explain.
4. Run the check command and the acceptance steps. Only a passing run lets you set `passes: true` for the entries it covered.
5. Run the `review` skill in a fresh context and fix what it finds.
6. Append a progress entry of at most five lines: done, verified with, open issues, next step.
7. List for the human every change to authentication, authorisation, personal data, secrets, migrations, CI or infrastructure, so they review those lines.
8. Stop. The human reviews, labels the PR `human-approved` if guarded paths changed, and merges.

If context per call passes 200,000 tokens before you finish, run `handoff` instead of pushing on.

