---
name: handoff
description: Close a session that has grown too long or changes topic, leaving a brief the next fresh session can start from. Use when context per call passes 200,000 tokens, before switching to an unrelated task, or at the end of a working day.
---

# handoff

1. Commit finished work with a conventional message. Leave unfinished work on its branch, never in the working tree alone.
2. Append to `progress.md` an entry of at most ten lines:
   - goal of the session and the acceptance entry or issue it served;
   - what is done and how it was verified;
   - what is left, as concrete next steps;
   - decisions taken and where they are recorded;
   - anything the next session must not repeat (dead ends, failing approaches).
3. If `progress.md` passes 5,000 tokens, move older entries to `docs/progress-archive.md`.
4. Give the human a one-paragraph brief to paste into a new session. It names the files to read first and the first command to run.
5. Stop. Do not continue in this session.

