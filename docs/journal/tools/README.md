# Tools that produced this journal

These are plain Python 3 scripts with no dependencies. Paths are hard-coded to this machine. Edit the constants at the top before reuse.

- `profile_transcript.py <file.jsonl>`: record types, sizes, models and compactions of one Claude Code transcript.
- `extract_digest.py <file.jsonl> <out.md> "<label>" [min_utc_timestamp]`: condensed chronological digest. It keeps the developer's prompts and the AI's visible prose in full, monitor events cut to 160 chars, outgoing cross-session messages up to 4,000 chars and incoming ones up to 1,000. Tool calls are reduced to their names, and tool output and reasoning are dropped. Compaction summaries go to `<out>.compactions.md`. Times are America/Chicago. Use the optional timestamp to drop the copied history at the start of a forked session (`2026-11-09T17:52` for df787a6f).
- `chunk_digest.py <digest.md> <prefix> <target_chars>`: splits a digest into chronological parts at developer-message boundaries. These parts are what the period journals were written from.
- `collab_stats.py`: computes the numbers in `stats/collaboration_stats.md`.

Transcript locations:
- `~/.claude_personal/projects/-Users-alon-projects-alpha-heroes/*.jsonl`: sessions cd, side, b1, e5 and the fork, plus `68ab2bd0…/subagents/`.
- `~/.claude/projects/-Users-alon/b608c016-….jsonl`: the 10-22 home-dir session.
- `~/.cursor/projects/Users-alon-projects-alpha-heroes/agent-transcripts/`: one Cursor exchange.
