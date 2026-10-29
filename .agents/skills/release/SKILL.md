---
name: release
description: Prepare a release and check the operational baseline before the human ships it. Use for every release and monthly for the operational checks.
---

# release

1. Confirm every entry in the release has `passes: true`, CI is green and, if AI behaviour changed, the eval report is current.
2. Write release notes from the merged PRs. For services and libraries, add the version and its changes to `CHANGELOG.md` following semantic versioning.
3. Check and report gaps:
   - backups and a restore test under a month old, if the project stores data;
   - alerts that page only on user-visible symptoms;
   - spend caps and a degraded path for paid APIs;
   - migrations: the expand step is deployed before code that needs it.
4. Stop. The human tags, releases and owns every production action. Agents hold no production credentials.

