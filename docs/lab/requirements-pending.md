# Requirements written in this branch, not yet registered in little-workshop

The session that built these phases had no little-workshop tools (no `workshop_*` MCP tools, no `lws`), so it could not
check `workshop_requirements_list`, create the requirements, record the sandbox test run, or start and finish the
branches. The ids below follow the user's statement that the next free id was REQ-000000027; **check them with
`workshop_requirements_list` before registering**, and if one is taken, renumber the tags in the named test file and in
this file together. Every test carries `@pytest.mark.requirement("REQ-...")`.

| Id | Phase | EARS text | Tests |
|---|---|---|---|
| REQ-000000027 | S1 shadow evaluation | When a forward score stored by a lab scorer has a matured label, the lab shall record its rank IC, its top-k paper return gross and net of one round trip and the equal-weight return of the same names using no price after the label's last session, shall never rewrite or back-fill a session, and shall write a weekly report pairing each model with the champion and with equal weight. | `tests/unit/test_lab_s1.py` |
| REQ-000000028 | X1 Qlib cross-check | When the x1-qlib-check service runs, the lab shall fit our trees and Qlib's LightGBM on identical point-in-time Alpha158 features, labels and purged folds of the development window and record their daily score correlation and paired rank-IC gap, falling back to plain LightGBM with the same parameters when Qlib cannot be imported, and recording which ran. | `tests/unit/test_lab_x1.py` |
| REQ-000000029 | B decision batch | When a decision-layer hypothesis service (H-D1 to H-D5, O1, O2) runs, the lab shall trade each pre-registered variant and the champion layer on the same real, planted and noise scores through the broker cost model, using only inputs known at each session and only spans inside the window the lab handed out, and shall record the 90 % block-bootstrap interval of the paired daily net-return difference in each span. | `tests/unit/test_lab_b_layers.py`, `test_lab_b_convex.py`, `test_lab_b_own.py`, `test_lab_hypothesis.py` |
| REQ-000000030 | T training batch | When a training hypothesis service (H-T1 to H-T6, O3, O4) runs, the lab shall score each pre-registered variant and the baseline trees by purged walk-forward on the development window with features, labels, weights and losses known at each session, and shall record the paired daily rank-IC difference with its overlap-corrected t and the Benjamini-Hochberg decision inside the family; every IC-like metric shall give about 0 for a constant or a noise score (O5). | `tests/unit/test_lab_t1.py`, `test_lab_t_feature.py`, `test_lab_t_loss.py`, `test_lab_t_ensemble.py`, `test_lab_hypothesis.py` |
| REQ-000000031 | I information | When an information study (H-I1, H-I2, H-I3) runs, the lab shall use an EDGAR filing or fact only from the session after it became public, shall read the EDGAR contact only from the lab's local config and refuse to fetch without one, shall send no more than 10 requests per second, and shall drop restricted issuers before any request. | `tests/unit/test_lab_info.py` |
| REQ-000000032 | F foundation models | When a foundation-model scorer (H-F1) runs, the lab shall score only the newest snapshot session from history up to that session, write the scores to its shadow log, never back-fill a session, and refuse any historical evaluation except a post-cutoff slice written to separate tables. | `tests/unit/test_lab_f1.py` |

R1 (notes) and G (summary) change no source and have no requirement.
