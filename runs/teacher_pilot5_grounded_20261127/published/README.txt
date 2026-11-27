Teacher grounding / targeted coarse scan: five-sample API verification (2026-11-27)

Tested source: 3b8543dcf51505244761f1d38aa702c80f1d88f5 on feature/bounded-refinement.
Same five full Train samples and qwen3.8-omni-flash as the previous bounded-refinement pilot.
Coarse chunks changed from 20 to 60 seconds alongside prompts and the candidate schema; this is not a single-variable ablation.
Budget: 64 observations; max output 8192 tokens; reasoning medium; one rollout per sample.

All five finished and generated Final; four complete and IELTS:183 partial.
138 / 106 / 193 / 183 / 176: 21 / 20 / 20 / 23 / 20 active observations.
135 actual API calls; 104 observations; 103 nonempty observation records.
No API failure, output truncation, format retry, exact repeated action, or invalid contextual citation.
All ended with candidate_pool_exhausted. Unit tests: 171 passed, 1 skipped.

Confirmed remaining issues:
1. IELTS:183 tail duration is 44.909002 seconds; candidate end 45 seconds exceeded it by 0.090998 seconds and was rejected, causing partial status.
2. IELTS:183 M14 text request [65,100] returned no segments; the intended concentration text is indexed around [103.4,137.1]. Those are unverified ASR retrieval times, not verified media alignment. The coarse text prompt omits the retrieval indexes but asks the model for times.
3. IELTS:106 Final treats clip-induced incomplete speech (M20 [227,240]) as negative evidence for conscientiousness.
4. Weak trait mappings persist; all five N=low and all 25 levels non-null. No accuracy claim or automatic SFT approval.

Start with report.json and quality_review.json. The two regression JSON files preserve executable check outcomes.
reviews/ has readable findings/actions/Final, and summaries/ has per-sample status and usage.
archives/ preserves every original per-sample JSON/JSONL file byte-for-byte, including full trajectories, raw model responses, rejected candidate details, usage, label-free diagnostic manifests, config and source hashes.
index.json records archive and file SHA256 values. Restore: tar -xJf archives/IELTS_138.tar.xz
logs/ includes launcher and unit-test logs.

All Final text sequences fit 32768 student positions (approximately 10.3k-11.6k); multimodal training memory was not tested.
No local 3B/7B inference rerun, formal SFT/RL, or Jev API call was performed in this test.
No complete human audit of the original media was performed. Original media, credentials and the master label-bearing manifest are excluded; local provenance paths are retained.
