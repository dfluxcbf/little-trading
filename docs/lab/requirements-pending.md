# Requirements written in this branch, not yet registered in little-workshop

The session that built these phases had no little-workshop tools (no `workshop_*` MCP tools, no `lws`), so it could not
check `workshop_requirements_list`, create the requirements, record the sandbox test run, or start and finish the
branches. The ids below follow the user's statement that the next free id was REQ-000000027; **check them with
`workshop_requirements_list` before registering**, and if one is taken, renumber the tags in the named test file and in
this file together. Every test carries `@pytest.mark.requirement("REQ-...")`.

| Id | Phase | EARS text | Tests |
|---|---|---|---|
| REQ-000000027 | S1 shadow evaluation | When a forward score stored by a lab scorer has a matured label, the lab shall record its rank IC, its top-k paper return gross and net of one round trip and the equal-weight return of the same names using no price after the label's last session, shall never rewrite or back-fill a session, and shall write a weekly report pairing each model with the champion and with equal weight. | `tests/unit/test_lab_s1.py` |
