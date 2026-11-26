Five-person teacher API pilot (2026-11-26)

Start with report.json for outcomes, SFT blockers and validation scope.
analysis.json contains detailed structural checks and offline label diagnostics.
reviews/ contains readable compact results, omitting duplicated prompt/history snapshots.
archives/ contains complete original JSON files, including full trajectories, rejected/partial records, original API responses and usage.
index.json lists archive and individual-file SHA256 checksums.
Restore, for example: tar -xJf archives/IELTS_138.tar.xz

All five sample attempts have stopped; no sample is still generating.
IELTS_176 is the original timed-out attempt. IELTS_176_resume is a separate manual continuation with exact prompt-checked local replay; both are retained.
No sample has been approved for SFT. No SFT/RL or Jev API call was performed.
Original source media, cached clips, review images and credentials are excluded.
Raw source metadata may contain local file paths; source media are not uploaded.
