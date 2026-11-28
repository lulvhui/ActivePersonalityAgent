Small Train-only Best-of-2 teacher candidate acquisition (tested 2026-11-27, published 2026-11-28)

TESTED SOURCE: c56ec929f5212319e06663d7dee31bf977d48c01.
Publication is appended after Claude commit 215bc6b; that newer curation/Final-RL code was NOT tested in this acquisition. Existing acceptance/filter results below refer to c56ec92 only.

Three distinct IELTS Train persons: 106,183,167. Two independent full-video trajectories each; qwen3.8-omni-flash; 64-observation budget; max output 8192; medium reasoning. No label in API input.
Six finished: five complete, one partial (167 r1 used global timestamps in local_time; rejected requests and raw outputs preserved).
145 real API calls, 107 observations, no API errors. No formal SFT/RL, Jev API, or local 3B/7B inference in this acquisition.

Start with report.json, code_review.json and candidate_index.json.
summaries/ contains each original per-run summary. reviews/ contains readable findings, actions and Final without repeated histories.
archives/ contains six complete original per-run JSON/JSONL sets: full trajectories, raw API responses and usage, rejected-candidate details, configs, source hashes and label-free manifests. Partial is preserved.
index.json records SHA256 hashes of every archive and enclosed file; all were verified byte-for-byte against the local originals.
Restore one original run: tar -xJf archives/IELTS_167_r2.tar.xz
Restore the five COMPLETE (unapproved) candidate files for downstream review: tar -xJf archives/complete_candidate_packet.tar.xz
This creates candidates/*.json, each containing a trajectory list; all retain teacher_accepted=false. Candidate-index relative paths resolve after extraction.

Only 167 r2 passes both protocol completeness and the old c56ec92 offline label-loss prefilter (0.20 <= 0.25).
106 r1/r2 losses: 0.35/0.35; 183 r1/r2: 0.35/0.30. 167 r1 loss is 0.25 but protocol is partial.
These are Train diagnostics, not held-out accuracy or evidence-truth judgments. Source and trait-semantic review is unfinished; zero trajectories approved for SFT. No unapproved SFT dataset was exported.

code_review.json separates Standards/Spec reviews (no deterministic code findings) from runtime model noncompliance. unit_tests.log: 174 passed, 1 skipped.
regressions.json checks the previous real 183 tail and transcript indexes. split_audit.json confirms no selected-person overlap with Dev/Test.
student_tokenization.json checks all 145 generated segments with the student tokenizer; all round-trip. Final text+target lengths 8159-11621 fit 32768. Multimodal expansion, GPU training memory and optimizer update were not tested.
acquisition_README.txt is the original local acquisition note, retained unchanged.

No credentials, raw media or model weights are uploaded. Local file paths remain as provenance.
