# Performance pass, 2026-10-29

Goal (the user): train fast; the RTX 5090 PC as the one GPU machine, fed by its own CPU and by other machines at home.
Nothing here changes what is learned or played: every change is the same computation done more cheaply (or measured
to be, below). Measured on the M4 MacBook Pro; the parts below measured while joint2 trained on it say so (a loaded
machine's processes land on the efficiency cores, so its CPU seconds are not comparable with a quiet one's).

## Where a generated battle's CPU went, and goes

4 generated actors, 2 CPU inference processes, 90 s (`bench_actors.py`, every process of the tree counted by kind), on
a quiet machine (2026-10-30: main d4f95bb, then these changes on it, alternately, twice each; the runs agreed within 1%):

| per decision, CPU ms | before | after | what changed |
|---|---|---|---|
| map building | 2.97 | 0.02 | `alpha_make_battle_map --serve`: one process loads the game's library once (`play.MapBuilder`) |
| VCMI client | 5.21 | 3.64 | the attack values' copy of the game state kept for a battle; records as int32 |
| trainer process | 0.94 | 0.33 | records parsed as views of the bytes (`data.npy_view`), int32 needs no narrowing |
| inference (CPU) | 8.80 | 8.63 | the trunk's attention fused, the BFS stops when its frontier is empty: little at a request's 1-2 decisions |
| **total** | **18.0** | **12.65** | the same 4 actors: 163 -> 194 decisions/s (their 2 inference processes now the limit) |

With the battle model on a GPU (the PC, and every worker through it) a decision costs a CPU 4.0 ms (was 9.2): the VCMI
client and the records. (On 2026-10-29, measured while joint2 trained: 27.6 -> 13.4 ms, 101 -> 178 decisions/s.)

## The joint run, before and after

`campaign_model.joint` resumed from the same copy of joint2 (epoch 5518, campaign version 390), evals off, on a quiet
machine (2026-10-30): 5 minutes to warm up, then 20 measured; main d4f95bb, then d9cb662 (these changes on it).

| | before | after |
|---|---|---|
| battle updates an hour | 126 | **156 (+24%)** |
| battle decisions trained a minute | 6,814 | 8,415 |
| an update's seconds waiting for battles | 4.36 | 2.28 |
| its training seconds: data / prepass / steps / publish | 23.28: 2.55 / 6.45 / 13.93 / 0.35 | 19.73: 1.01 / 6.07 / 12.25 / 0.40 |
| campaign decisions trained (updates) | 48,910 (5) | 47,238 (4) |
| campaign games played (their decisions) | 48 (138,715) | 47 (138,313) |
| cores in use | 6.76 | 6.53 |

- On the Mac the GPU is the limit: training is 88% of a battle update, on the M4's GPU beside the campaign learner's
  updates, which kept their pace (26 ms a decision either way). The battle side gained; the campaign side lost nothing.
- The campaign learner trains on about a third of what its games play (25 of the 48 games dropped as too old,
  `CampaignLearner._take`): on this machine its games are not its limit.
- Campaign battles were 0.093 of each battle update before, 0.060 after (the recipe asks 0.7): the same few campaign
  battles (4 in a 26-day game on average, 9 decisions each) over 24% more updates. None are lost: in 6 campaign games
  with ALPHA_BATTLE_ATTACK_CHECK=1 every battle a client logged reached the record server, and the kept copy's attack
  values equalled a new copy's in every checked decision (137).
- Exact after the merge: the attack check on 560 generated battles (39,140 decisions) found no difference in either
  mode (2-3 a run before VCMI's target order fix, e1be37e4a), and 32 greedy battles replayed twice recorded the same
  107,819,712 values.

## The costly small things

- `BattleSimulation::copyOf` in `encodeAttackValues`, every decision: a copy of the whole game state through savegame
  serialization, 37% of a battle client's CPU (more in campaign games: the copy is the adventure map). Now one copy a
  battle (`AttackValueCache`), every attack still played from the battle as it is. On the same 30 battles: 6.16 -> 2.26
  ms of attack values a decision. `ALPHA_BATTLE_ATTACK_CHECK=1` values every decision on a new copy too and logs any
  difference: 2 of 36,879 decisions differed by 1-15 gold, and 3 of 36,530 between two new copies (=2), the same cases:
  the engine hits an area attack's units in the order of their addresses (`BattleActionProcessor::makeAttack`, a
  `std::set<const CStack *>`), and the simulation's alternate rounding of mean damage depends on that order. Not the
  kept copy's; probably the ~1% of evals that did not replay exactly. Fixed in VCMI (e1be37e4a: such targets in unit id
  order): 0 differences since, in either mode.
- A new `alpha_make_battle_map` process for every generated battle: ~0.2 s of CPU (the library's load), ~17% of a
  generated actor's CPU. Now ~2-4 ms a map.
- AlphaMapAI compressed each turn's records at zlib level 6 (~38% of its busy CPU), and the trainer decompressed them
  and compressed each decision again at level 1. Now each decision at level 1 in C++ ("AMZ2", records.py), kept as it
  comes; the trainer asks for it (ALPHA_MAP_RECORD_FORMAT=2), so an older trainer still gets "AMZ1" (at level 1 now):
  the code and the build can be updated in either order. A 10-day campaign game's client: 9.51 -> 6.01 ms of CPU a
  decision.
- Battle records as int64 (0.68 MB a decision): the trainer read every array to narrow it. Now int32, as the inference
  wire already sends them (out-of-range inputs were already refused there).

## The learners (GPU)

- Trunk attention: the relation bias r[h, relation(i, j)] is r1 + a (|s_i| |s_j|) + b (s_i s_j) with s the token's side,
  so it folds into two more q/k dims and `scaled_dot_product_attention` runs it: no [B, heads, T, T] bias or mask is
  made (the same function to float rounding, 6e-6 in log-probabilities; gradients 1e-6 relative). CPU forward+backward
  of a block 2x faster; Apple GPU training step -7% at batch 64; on CUDA the fused kernels should gain more (not measured).
- An update's inputs move to the GPU as int32 (1.0 GB a 3,200-decision update, was 1.9 GB) and are widened a minibatch
  at a time (`data.widened`); collate 1.05 -> 0.52 s.
- Battle inference server on a GPU: one copy to the device and one back a batch (was one an input and one an output).
- Campaign update: one sync a step (was six `.item()` a micro-batch and the gradient norm); on CUDA the inputs are
  pinned in the prefetch thread and copied without blocking (each pageable copy waited for the GPU). The campaign's
  inference server on CUDA when there is one (`campaign.inference_device`, "auto"; it was always the CPU, 5-10 ms of a
  core a decision). `micro_batches = 1` for the 5090 (the same gradient as 4 parts).
- Tried and left out: `torch.compile` (CPU inference only 27% faster for a 63 s compile a batch size; training's
  gradients differed from eager by up to 110% relative in a quick CPU test: not trusted without CUDA to test on).

## Several machines

`battle_model/remote.py` (the master's hub) and `battle_model/worker.py`: README "On several machines". A worker plays
generated battles and campaign games with the master's inference servers over TCP (zlib: a battle decision's request
293 KB -> 2.8 KB, its reply 105 KB -> 1.0 KB), and pushes its checked, compacted records and finished games to the
master; the master's state (model version, league, pause for evals, room in the buffer) comes back every second.
Pushed rather than asked for game by game: the buffer's backpressure ("room") already says when to play, and nothing
waits for a request's round trip. Tested on one machine (a master and a worker over 127.0.0.1): a battle run (the
learner's wait 10 s -> 0.1-4 s an update with the worker's two actors) and a joint run (the worker's 8 campaign games
and 189 battles trained on, 0 failures, both ends stopping cleanly).

## Not done (recommendations)

- CPU inference is now 68% of a CPU actor's cost (8.6 of 12.65 ms a decision) and did not get cheaper: a request is 1-2
  decisions, where the fused attention gains little. The legal-entries-only action heads (next) are the way to cut it;
  on the PC the battle model infers on the GPU and this does not arise.

- The battle model scores all 28 x 187 melee-from and 70 x 187 spell-hex entries, of which ~3 and ~7 are legal on
  average (70% and 92% of decisions none): the action heads are 45% of a training-size forward. Scoring only the legal
  entries gives the same outputs (the rest are masked) for a fraction of the work, but needs shapes that change with the
  batch (bad on Apple's GPU, a sync on CUDA): worth it for CPU inference, less on the 5090.
- The campaign model: 30 ms a decision forward+backward on the CPU, two thirds of the forward in the map encoder and the
  hand-made features, computed three times a decision an update (the KL anchor and two passes). Its parameter-free
  features could be computed once, as the battle model's `facts` are; they are mixed with trainable weights now.
