# Datathon 26 — ForteBank Cash Discipline AI

Local visual observations from cash-desk video, with real presentation timestamps
and evidence JPEG paths. ML proposes candidates; a separate Rule Engine evaluates
them. No face recognition, cloud AI, legal conclusions or guaranteed model accuracy.

Participant 2 implements `app/vision/`; shared Pydantic contracts are in
`app/schemas.py`. API/worker/UI and legal rules are separate team responsibilities.
See [team](docs/TEAM.md), [architecture](docs/architecture.md) and [agent rules](AGENTS.md).

This initial commit contains the common foundation. Vision implementation follows
on `feature/vision-pipeline`.
