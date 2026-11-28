Small teacher candidate acquisition, source c56ec929f5212319e06663d7dee31bf977d48c01

3 distinct IELTS Train persons (106,183,167), 2 independent rollouts each; full videos; qwen3.8-omni-flash.
6 finished, 5 complete, 1 partial (167 r1 recording/local-clock mix-up).
145 API calls, 107 active observations. No formal SFT/RL or Jev API call.

Start with report.json and candidate_index.json.
candidates/ contains the 5 complete trajectories, one JSON list per file, all teacher_accepted=false.
IELTS_*_r*/ contains each original trajectory, raw responses, usage, config, source hashes and label-free input. The partial trajectory is retained there.
Best-of-2 groups are in report.json; no winner is automatically approved.
Only IELTS_167_r2 passes both protocol completeness and the existing offline label-loss prefilter (0.20 <= 0.25); source and trait-semantic review remain incomplete.
106 r1/r2 label loss: 0.35/0.35; 183 r1/r2: 0.35/0.30. These are Train-only diagnostics, not held-out accuracy.
Labels are read only after generation for that diagnostic; they never enter teacher prompts.

code_review.json separates independent Standards and Spec reviews, both without deterministic findings, from the runtime model-compliance issue.
unit_tests.log: 174 passed, 1 skipped. regressions.json uses the prior real 183 input to check tail normalization and transcript indexing.
split_audit.json confirms no selected-person overlap with Dev/Test.
student_tokenization.json checks 145 genuine generated segments with the student tokenizer; all round-trip. Final text sequences are 8159-11621 tokens. Expanded multimodal positions and optimizer updates are not tested.

No approved SFT dataset has been exported. Existing export refusal for unapproved trajectories was verified.
Review IELTS_167_r2 sources and trait linkage first; do not remove true contrary behavior to improve label agreement. Before expanding acquisition, improve explicit local-clock bounds in coarse input rather than silently guessing how to repair recording-clock requests.
