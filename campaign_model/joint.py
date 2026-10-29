"""Campaign training alongside the battle model's (the user, 2026-10-27): one run, two models, each with its own
optimizer; the campaign's updates in a process of their own (learner.UpdateProcess) on the GPU next to the battle
learner's.

	.venv/bin/python -m campaign_model.joint --run joint --hours 12
	.venv/bin/python -m campaign_model.joint --run joint --resume              # goes on after a stop
	.venv/bin/python -m campaign_model.joint --run joint --set campaign.games=6 generated_actors=6
	.venv/bin/tensorboard --logdir runtime/train                                # battle: <run>/tensorboard/run, campaign: .../campaign

The battle side is v15-window's recipe (battle_model/train.py: generated battles with hard cases among them, the
self-play league, evaluations against BattleAI and MMAI every hour), going on from joint1's best checkpoint
(BATTLE_FROM: 2840, every eval set positive; the plan of 2026-10-28 continues the battle model, it does not restart
it), with the campaign games' battles as its real battles (real_share of each update, their advantages normalised
apart from the generated ones') instead of Nullkiller2's games: the generated ones keep it from forgetting the rest.
The campaign side (learner.py, config.py): campaign.games (6) self-play games at once, an update per ~8k decisions of at
least 4 games, an evaluation against Nullkiller2 every hour (evaluate.py).
"""
import argparse
import signal

from battle_model.train import ROOT, TrainConfig, train, with_overrides

BATTLE_FROM = ROOT / "runtime" / "saved" / "battle_joint1_02840.pt"  # joint1's 2840 (+0.076, every set positive), a read-only copy; runtime/saved/models: joint1's league snapshots up to it (no eval baselines: the reward changed with the war machines)
RECIPE = (
	"generator=true", "actors=true", "generated_actors=8", "real_games=0", "real_share=0.4",
	"eval_scenarios=24", "eval_neutral_scenarios=12", "eval_hard_scenarios=24", "eval_real_scenarios=24", "eval_castle_seeds=16",
	"eval_minutes=60", "seed=1", "campaign.games=6", "campaign.inference_processes=2", "campaign.eval_minutes=60",  # (joint2, 2026-10-28: 8 games and 10 actors swapped; 6 and 8 fit in 24 GB)
	"advantage.credit=window",  # (the default: the battle model's linear round window since 2026-10-26, no GAE lambda)
)


def stop_on_signals():
	"""SIGTERM and SIGINT end the run as Ctrl-C does (its games, servers and VCMI processes closed): a run started in
	the background from a script has SIGINT ignored, so kill -INT did nothing and kill -TERM left its children."""
	def stop(signum, frame):
		raise KeyboardInterrupt(f"signal {signum}")

	signal.signal(signal.SIGTERM, stop)
	signal.signal(signal.SIGINT, stop)


def main():
	stop_on_signals()
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--run", default="joint")
	parser.add_argument("--hours", type=float, default=0.0, help="stop after this many hours of training (0: until stopped)")
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--battle-from", default=str(BATTLE_FROM), help="the battle model's starting checkpoint (a new run)")
	parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="override TrainConfig fields after the recipe; campaign.NAME for the campaign side")
	args = parser.parse_args()
	# (epochs go on from the starting checkpoint's: without hours, no epoch limit)
	config = TrainConfig(run=args.run, run_hours=args.hours, epochs=10**9, init_from=args.battle_from)
	train(with_overrides(config, RECIPE + tuple(args.set)), resume=args.resume)


if __name__ == "__main__":
	main()
