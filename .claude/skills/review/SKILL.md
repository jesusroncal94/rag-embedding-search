---
name: review
description: Review a diff in a fresh context against its spec, acceptance entries and the project rules, before the human sees it. Use at the end of every build-slice and before opening a pull request.
---

# review

Run this in a context that did not write the code: a subagent or a new session. Read only what the review needs.

1. Read `AGENTS.md`, the spec, its acceptance entries and the diff (`git diff main...HEAD`).
2. Check, in this order:
   - every acceptance step has a test that would fail without the change;
   - no existing test, acceptance criterion or guarded path was changed;
   - no secret, key, personal data or prompt reaches code, logs or commits;
   - new dependencies exist in the official registry and are justified;
   - authorisation: each new endpoint, table or file path has a negative test;
   - the diff does nothing the spec did not ask for.
3. Report findings as a short list: file, line, problem, why it matters. Say "no findings" when there are none; do not pad.
4. Do not fix anything yourself. The builder fixes; the human decides on disagreements.

