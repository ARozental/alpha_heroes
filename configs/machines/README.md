# Machine configs

How many games, actors and servers a machine runs: its best sizes for each kind of training, measured there. Never what
is learned (the recipe: learning rate, rewards, eval sets) - that stays on the command line or in the recipe
(`campaign_model/joint.py` RECIPE), so the same run can be compared across machines.

```
configs/machines/<machine>/<training>-<case>.toml
  training: battle (battle_model.train, actors=true) | campaign | joint (campaign_model.joint)
  case:     gpu-free (nothing else on the GPU) | gpu-shared (another training run on the GPU, e.g. Zonkey)
```

Use one with `--machine`; it is applied after the recipe and before `--set`, so `--set` still overrides it:

```sh
.venv/bin/python -m battle_model.train --run NAME --machine configs/machines/rtx5090/battle-gpu-shared.toml --set learning_rate=3e-5 ...
.venv/bin/python -m campaign_model.joint --run NAME --machine configs/machines/rtx5090/joint-gpu-free.toml
```

A file is TOML: top-level keys are `TrainConfig` fields (`battle_model/train.py`), a `[campaign]` table sets
`campaign.*` (`campaign_model/config.py`). Its comment says the machine's specs, the case, what was measured (update
seconds, the learner's wait, peak memory, the bottleneck) and when - or that it is an estimate not measured yet.

Measuring a case (Linux; the scripts are in `runtime/`, which git ignores - copy them from the rtx5090 machine):
`runtime/measure.sh` starts a short run as a systemd unit, `runtime/memwatch.sh` samples its memory and stops it before
the kernel's OOM killer would act, `runtime/progress_table.py` and `runtime/campaign_table.py` show where an update's
seconds go (`wait_seconds` high: the battles are the bottleneck; `train_seconds` high: the learner).

| machine | specs |
|---|---|
| `rtx5090` | RTX 5090 32 GB, Ryzen 9 9900X (12 cores / 24 threads), 31 GB RAM, Windows 11 + WSL2 (20 GB) |
| `mac-m4` | MacBook Pro M4, 24 GB unified memory (to add: its sizes so far are in campaign_model/joint.py's RECIPE comment) |

`campaign-*` (the campaign model alone) has no files yet: training has no campaign-only mode. A run with
`generated_actors = 0` still trains the battle model, on campaign battles only, and each battle update then needs the
battles of >= 8 campaign games (`game_share` 0.125); its learner waits for them and stops after an hour without.
