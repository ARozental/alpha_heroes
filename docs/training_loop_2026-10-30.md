# The training loop rebuilt around a data loader (2026-10-30/31)

The user's directive: make training fast; the GPU's time on training, or every core busy on both machines; the games
produced by workers of an `IterableDataset` + `DataLoader`, never more than the learners take. This note says what was
built, what was measured on the RTX 5090 PC (Windows 11, WSL2) with the MacBook Pro M4 beside it, and what remains.

## What was built

- **`battle_model/supply.py`**: two `IterableDataset`s (generated battles, campaign games) in `DataLoader`s
  (`in_order=False`, forkserver workers). A worker plays one game at a time, here while this machine is under its
  ceiling (`generated_actors`, `campaign.games`) and above a memory floor, else on a remote machine's slot, else it
  waits. It turns the records into training arrays (`prepare`: `battle_examples` + `collate`, the big tables sparse)
  and yields them; a campaign worker yields each battle as it ends, then the game. The trainer pulls what it will
  train on: generated battles by the loader's prefetch and `ActorPool.generated_room` (`ahead` updates' worth),
  campaign games by permits (`Supply.allow`: `ahead` updates' worth minus finished and in flight).
- **Remote machines are game servers** (`worker.py --slots N --local [--serve PORT]`): N connections held on
  `remote_port + 100`, one game each on request; the worker's own inference servers get the master's weights from the
  hub and, with `--serve`, also answer the master's games (`inference_endpoints`).
- **One trainer process**: campaign updates between battle updates in the trainer's thread (`campaign.learner="turns"`),
  whichever has data first; both models' inference in one server process when it is here (`InferenceServer(families=...)`).
  Evals run in threads beside training, the evaluated weights pinned as `"eval"` on the servers.
- **`campaign_model/labeler.py`**: the labels a game needs that no model forward gives (`label`, `growth_labels`,
  `end_labels`, `end_steps`, `foresight.labels_of`) computed per game as it arrives, in two processes; `step()` takes
  them (recomputed inline past `MAX_LABEL_STALENESS` versions; `label_staleness` logged).
- Campaign inputs go to the GPU as the wire bytes and are cut and converted there (`records.inputs_on_device`), decoded
  ahead in `DECODE_THREADS`; the day-start forwards use the same prefetch. `ppo_update` logs where its time goes
  (`prepass_seconds`, `wait_seconds`, `sync_seconds`, `cpu_share`).

## What was measured (the PC, `runtime/bench_mixed.py` and the live run)

| what | measured |
|---|---|
| campaign step, 128-decision micro-batch, GPU alone, synthetic states | 146 ms eager |
| the same in the live update (real states, eager) | ~300 ms; the trainer's thread 52–64% on the CPU: launch-bound |
| a second GPU process (inference) beside the trainer | the trainer's steps −54% (−40% with a 10–15 ms batch wait): WDDM time-slices by demand |
| inference as a thread in the trainer | the trainer 12x slower (the interpreter lock) |
| CPU campaign inference | PC 38 decisions/s a core under load; M4 ~90 |
| battle step, 256 decisions | 36 ms eager, 25 ms graphed |
| host→GPU copy under WSL | 27 GB/s pinned: not a limit |
| campaign update, 35k decisions | joint5 before: ~140 s with ~27 s of labels in the trainer's thread and games 90% dropped; now ~80–110 s eager (prepass graphed 19–24 s), labels off the thread |
| trainer RSS | +3–5 GB on a process's first update (CUDA warm-up, pinned cache), ~7–10 GB steady |

Inference placement that came out of it: the Mac's cores serve the campaign model over the LAN (9 processes; WSL
reaches the Mac directly, Wi-Fi 14–61 ms a round trip), the PC's CPU serves the battle model (3 processes), nothing but
the trainer on the GPU.

## Settled on 2026-10-31 (b1's recipe calls)

- `campaign.passes = 2`: the second pass is fully used (the KL stop never cut it, kl 0.004–0.015 against a 0.0225 line);
  the trainer is then the limit and the permits hold back ~10% of game starts, which b1 judged an acceptable price.
  Revert to 1 if the campaign eval trend over 3–4 versions is flat or down against v47–v59, or if `stopped_pass` shows
  the KL stop cutting pass 2. Updates kept at 32,768 decisions of 20+ games (fewer games an update is what the user
  fears most); the prepass kept (the anchor matters more with staler games).
- `ahead = 3`: with 2 the permits emptied the pipeline during each ~150 s update and the trainer waited a game's
  duration after it (GPU 62% against 74%). Games waiting two versions get their labels redone in the labeler
  (`learner._relabel_waiting`).
- `campaign.micro_batches = 4` (128 decisions): 256 filled the card (18 GB of live tensors plus the allocator's reserve
  and the graphs' pools) and the step was 1.5x slower a decision while `nvidia-smi` read 94%: judge the GPU by
  decisions/s, not by its utilisation figure.
- Ten-minute readings of the final configuration (24 games, 8 actors, 4-process evals): GPU mean 70%, 64% of samples at
  or above 80%, campaign updates 100–153 s back to back, min free RAM 4.9 GB (24 games is the RAM limit), ~0 game
  failures an hour. What is left of the GPU's idle time: the decode waits (~5%, `DECODE_THREADS`), the eager loss
  launches between graph replays (~5%: capturing the loss with the step would take them), the battle updates between
  campaign updates (~10%, GPU-busy themselves but with CPU-side assembly).

## What limits the GPU now

The trainer waits for data most of the time: the battle learner gets ~250 decisions/s from 10 actors and needs ~2.5 s
of GPU per update; the campaign pipeline is bursty (an update takes ~35k decisions, the games in flight hold about one
update's worth, so each update waits a game's duration). More games are bounded by the PC's RAM (27 GB: trainer
~7 GB, 34 loader workers ~6.6 GB real, VCMI ~0.15 GB a game) and the Mac's cores (its 9 campaign servers at ~70%).
The levers left are the recipe's: `campaign.passes` 1 → 2 (twice the GPU work per game), smaller and more frequent
campaign updates; and hardware: an Ethernet cable (every decision pays the Wi-Fi round trip), RAM in the PC, or Linux on
it (no WDDM time-slicing, so inference could share the GPU).

## Incidents worth remembering

- `runtime/memwatch.sh` stops the run under 1.5 GB available: it did once (44 workers + trainer + VCMI). Workers now
  `gc.freeze()` what they inherit, trim the heap after each game, and start no game under 4.5 GB (`MEMORY_FLOOR_GB`).
- A restart of the run killed every game on the Mac's servers (they lived with the hub connection); they now persist
  across reconnects. A stale `ssh -N` held the tunnel ports after a restart; `runtime/mac_slots.sh` keeps running
  through a restart and must be started with no old tunnel alive.
- A `Future` stored on a game made it unpicklable for the labeler; futures live in a side dict.
- The training CUDA graph (`graphs.py`) needs `allow_unused_input=True` and no eager autograd graph alive at its first
  capture (an AccumulateGrad node referencing the default stream fails the capture).
- VCMI's random map generator is not deterministic by seed: the eval maps are kept with the code (`eval_maps/`).
