# Collaboration statistics: how the project was built

Computed on 2026-11-12 from the raw Claude Code transcripts (JSONL) and the git history by `tools/collab_stats.py`.
Times are America/Chicago. "Developer prompt" means a message the developer typed. Monitor events, cross-session messages, slash commands, interrupts and system text are counted separately, and consecutive duplicates are collapsed. "AI turn" means one model API response (one assistant message id).

## 1. Sessions

| name used in the journal | transcript id | role | span (local) | developer prompts | AI turns | tool calls | compactions | interrupts | cross-session msgs in / out | models (AI turns) |
|---|---|---|---|---|---|---|---|---|---|---|
| `cd` (alpha-heroes-cd) | a62ff69f | founding: VCMI setup, battle model, first training, infra | 10-21 21:47 → 10-29 07:47 | 170 | 3,480 | 3,344 | 9 | 3 | 0 / 0 | Opus 5.5 (all) |
| `home` | b608c016 (in `~/.claude`) | side session from the home dir that touched the repo | 10-22 11:10 → 14:39 | 12 | 62 | 66 | 0 | 1 | 0 / 0 | Opus 5.5 |
| `side` | fd6b4705 | short side session (empire value question) | 10-27 00:46 → 02:04 | 1 | 63 | 67 | 1 | 0 | 0 / 0 | Opus 5.5 |
| `b1` (alpha-heroes-b1) | 68ab2bd0 | main science: campaign AI, Φ, architecture, PC run | 10-27 00:54 → 11-12 08:50 (live) | 280 | 13,241 | 12,638 | 30 | 15 | 120 / 230 | Opus 5.5 13,228; Fable 5.1 13 |
| `e5` (alpha-heroes-e5, later -88) | 31529395 | perf, infra, reviews/audits, deploys, the PC | 10-29 21:39 → 11-12 08:50 (live) | 135 | 3,331 | 3,538 | 23 | 5 | 171 / 192 | Opus 5.5 773; Fable 5.1 429; Fable 5.2 2,129 |
| `fork` | df787a6f | fork of b1 (2026-11-09): presentations | 11-09 11:53 → 11-12 08:43 | 14 | 262 | 318 | 5 | 2 | 0 / 0 | Opus 5.5 |
| subagent | agent-af5e… (under b1) | one 8-minute subagent (2026-10-31) | 10-31 14:15 → 14:23 | 0 | 40 | 39 | 0 | 0 | 0 / 0 | Opus 5.5 |
| Cursor agent | 63ae246c (in `~/.cursor`) | one small Cursor-IDE exchange (2026-10-24) | 10-24 | – | – | – | – | – | – | – |
| claude.ai chat | (shared link) | the planning chat that set the goal and stack, pasted in on 10-21 | before 10-21 | – | – | – | – | – | – | – |

**Totals:** 612 developer prompts, ~20,480 AI turns, ~20,000 tool calls, 68 context compactions, 422 cross-session messages sent (b1 230, e5 192) and 291 received (b1 120, e5 171). Some sends went to stale socket addresses.
**Tokens:** 26.1M output, 65.0M cache-write, 9,376M cache-read and 0.06M uncached input. Tokens per session (output / cache-read): b1 16.0M / 6,641M; cd 3.9M / 1,545M; e5 5.4M / 1,076M; fork 0.6M / 85M.
Cost depends on the developer's plan. To estimate an API-equivalent cost, apply current list prices to these volumes. Almost all input was served from the prompt cache, because the sessions are very long and resumed.

**Tools:** about 85–90% of tool calls were `Bash`. For b1 that was 10,804 of 12,638: tests, git, ssh/WSL to the PC, monitoring scripts, and many code edits done through scripts. Edit+Write: cd 137, b1 589, e5 195. `Monitor` (background watchers on training logs): b1 780, cd 96.
The dev sessions almost never used subagents (one in total). Each was one long-lived agent that coordinated with the others only through cross-session messages (`SendMessage`).

## 2. Activity per day

| day | developer prompts | cd / b1 / e5 / fork prompts | AI turns cd / b1 / e5 | tool calls | cross-session msgs sent | commits |
|---|---|---|---|---|---|---|
| 10-21 | 27 | 27 / – / – / – | 179 / – / – | 177 | 0 | 6 |
| 10-22 | 35 | 23 / – / – / – (+12 home) | 353 / – / – | 444 | 0 | 16 |
| 10-23 | 39 | 39 / – / – / – | 898 / – / – | 892 | 0 | 33 |
| 10-24 | 27 | 27 / – / – / – | 648 / – / – | 646 | 0 | 13 |
| 10-25 | 13 | 13 / – / – / – | 371 / – / – | 314 | 0 | 15 |
| 10-26 | 23 | 23 / – / – / – | 638 / – / – | 560 | 0 | 30 |
| 10-27 | 20 | 14 / 5 / – / – (+1 side) | 233 / 376 / – | 666 | 0 | 3 |
| 10-28 | 31 | 3 / 28 / – / – | 122 / 1,037 / – | 1,139 | 0 | 0 |
| 10-29 | 26 | 1 / 24 / 1 / – | 38 / 789 / 270 | 1,082 | 0 | 14 |
| 10-30 | 48 | – / 24 / 24 / – | – / 785 / 635 | 1,502 | 34 | 32 |
| 10-31 | 32 | – / 26 / 6 / – | – / 985 / 70 | 1,032 | 9 | 57 |
| 11-01 | 20 | – / 18 / 2 / – | – / 790 / 28 | 766 | 0 | 36 |
| 11-02 | 28 | – / 25 / 3 / – | – / 577 / 30 | 600 | 1 | 11 |
| 11-03 | 15 | – / 11 / 4 / – | – / 586 / 27 | 558 | 0 | 16 |
| 11-04 | 27 | – / 26 / 1 / – | – / 1,116 / 9 | 1,074 | 2 | 19 |
| 11-05 | 44 | – / 31 / 13 / – | – / 1,536 / 64 | 1,662 | 0 | 34 |
| 11-06 | 18 | – / 10 / 8 / – | – / 660 / 35 | 656 | 0 | 6 |
| 11-07 | 31 | – / 31 / 0 / – | – / 808 / 15 | 752 | 3 | 14 |
| 11-08 | 9 | – / 7 / 2 / – | – / 296 / 17 | 276 | 0 | 3 |
| 11-09 | 23 | – / 9 / 8 / 6 | – / 301 / 307 | 809 | 11 | 8 |
| 11-10 | 14 | – / 1 / 6 / 7 | – / 563 / 844 | 1,523 | 81 | 31 |
| 11-11 | 58 | – / 4 / 54 / – | – / 1,492 / 789 | 2,212 | 202 | 64 |
| 11-12 (to 08:50) | 4 | – / – / 3 / 1 | – / 544 / 191 | 668 | 79 | 5 |

Notes:
- **Who the developer talked to shifted over time.** cd took everything until 10-27, then b1 took most prompts. From 11-09 the developer increasingly directed e5. On 11-11 e5 got 54 of 58 prompts while b1 worked mostly from e5's messages: 202 cross-session messages that day.
- **Commits by hour of day (466 total):** about 77 were made between 00:00 and 06:59, mostly autonomous overnight work.

  | hour | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | commits | 21 | 17 | 7 | 14 | 7 | 5 | 6 | 13 | 13 | 31 | 27 | 16 | 27 | 22 | 19 | 25 | 24 | 28 | 26 | 22 | 14 | 22 | 35 | 25 |

- **Commit co-author tags:** 427 Opus 5.5, 32 Fable 5.2, 2 Fable 5.1. All 466 commits are authored by the developer's git identity.

## 3. Long autonomous stretches

These are stretches of at least 3 hours between developer prompts, with more than 50 AI turns in between. They usually happened overnight, with the AI monitoring and changing the training run on its own.

| session | start | hours without a developer prompt | AI turns in between |
|---|---|---|---|
| b1 | 11-09 22:34 | 19.6 | 382 |
| b1 | 11-10 18:11 | 17.3 | 658 |
| b1 | 11-06 16:48 | 14.0 | 138 |
| e5 | 10-30 22:46 | 14.1 | 113 |
| fork | 11-09 17:02 | 14.7 | 57 |
| b1 | 11-08 20:17 | 11.5 | 68 |
| b1 | 11-07 21:24 | 10.9 | 346 |
| cd | 10-25 22:20 | 10.7 | 320 |
| e5 | 11-10 13:29 | 10.5 | 445 |
| b1 | 11-01 20:38 | 10.4 | 94 |
| e5 | 10-29 21:46 | 10.3 | 270 |
| cd | 10-24 22:27 | 9.8 | 144 |
| b1 | 10-31 22:37 | 9.6 | 363 |
| b1 | 10-29 22:43 | 9.3 | 173 |
| b1 | 11-02 22:12 | 9.3 | 265 |
| b1 | 11-05 22:08 | 9.3 | 367 |
| b1 | 11-04 21:59 | 9.1 | 531 |
| e5 | 11-11 23:26 | 8.5 | 175 |
| cd | 10-22 23:52 | 7.7 | 223 |
| cd | 10-27 01:22 | 7.6 | 159 |

The first was the night of 10-22. The developer said "I'm going to sleep you will continue to monitor and you may try different things to improve … no experiment should run for over 1000 epochs, I'll get back in 8 hours". From then on, overnight autonomy was the normal mode. Many collapses, rollbacks and "lucky eval" misreadings in the period journals happened inside these stretches.

## 4. Code size on 2026-11-12

| part | size |
|---|---|
| Python, non-test (`battle_model/`, `campaign_model/`, `game_values/`) | ~28,600 lines |
| Python tests | ~13,200 lines, 585 `test_` functions |
| C++ AI plugins (`agents/`: AlphaBattleAI, AlphaMapAI, UpLeftAI, …) | ~14,300 lines |
| VCMI engine fork | 45 project commits since 2026-10-20; 60 files changed, +1,061 / −1,206 lines vs the pre-project base (including removed upstream CI files) |
| Git | 466 commits in 23 days (busiest: 64 on 11-11, 57 on 10-31) |
| Docs and memory | 17 docs (~280k chars) and 79 memory notes (~560k chars) written by the AI sessions |

## 5. Corpus behind this journal

- Raw transcripts: about 527 MB of JSONL in `~/.claude_personal/projects/-Users-alon-projects-alpha-heroes/`, plus about 1.3 MB in `~/.claude/projects/-Users-alon/`. Most of the bytes are tool output and model reasoning.
- Condensed digests in `transcript_digests/`: about 5.2M chars, with the developer's prompts in full, the AI's visible prose in full, cross-session messages, subagent briefs and monitor events. Tool output and reasoning are left out.
- Claude Code deletes transcripts about 30 days after their last change by default (`cleanupPeriodDays`). If the raw record should outlive the sessions, archive the JSONL files or raise that setting.
