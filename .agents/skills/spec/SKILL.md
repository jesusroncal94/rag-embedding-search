---
name: spec
description: Turn a feature or change into a risk-scaled spec and acceptance entries for human approval. Use before building anything larger than a one-line fix.
---

# spec

1. Size it. Full spec when the change touches the data model, authentication, authorisation, money or personal data; quick spec otherwise; no spec for a one-line fix.
2. Ask the human only what changes the work: why (link an assumption or issue), the behaviour, what is out of scope.
3. Write `docs/specs/<ID>-<slug>.md` from `docs/specs/_template.md`. Keep implementation detail out of the spec; it decays.
4. Propose entries for `acceptance/features.json`: concrete steps a test can run, including at least one negative step (another user, bad input, missing permission).
5. If a hard-to-reverse choice is needed, draft an ADR in `docs/adr/` with status `proposed`.
6. Stop. The human approves the spec, the acceptance entries and any ADR. Building starts in a fresh session with `build-slice`.

