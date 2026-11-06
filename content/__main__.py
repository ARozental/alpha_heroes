"""The content match from the command line (docs/hota_transfer_plan.md, Part 1).

	python -m content export <user dir> <out.json>             the engine's content as that user dir's mods load it
	python -m content check <vocabulary.json> <engine.json>     the readable diff; exit 1 unless clean ([--strict])
	python -m content merge <vocabulary.json> <engine.json> <out.json>   the vocabulary grown append-only

A vocabulary or engine file: a vocab.json with the exporter's "content" section, or a content file itself."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import is_prefix, load, match, merge, save

ROOT = Path(__file__).resolve().parent.parent


def export(user_dir, out):
	"""Runs the exporter (build/bin/alpha_export_vocab) with VCMI_USER_DIR=user_dir, keeps its content section."""
	with tempfile.TemporaryDirectory() as tmp:
		full = Path(tmp) / "vocab.json"
		subprocess.run([str(ROOT / "build" / "bin" / "alpha_export_vocab"), str(full)], env={**os.environ, "VCMI_USER_DIR": str(Path(user_dir).resolve())}, check=True, capture_output=True)
		save(load(full), out)


def main(argv):
	strict = "--strict" in argv
	args = [a for a in argv if a != "--strict"]
	if len(args) == 3 and args[0] == "export":
		export(args[1], args[2])
		print(f"wrote {args[2]}")
		return 0
	if len(args) == 3 and args[0] == "check":
		result = match(load(args[2]), load(args[1]))
		print(result.report())
		clean = result.clean(strict)
		print(("clean" if clean else "NOT CLEAN") + (" (strict)" if strict else ""))
		return 0 if clean else 1
	if len(args) == 4 and args[0] == "merge":
		old, engine = load(args[1]), load(args[2])
		grown = merge(old, engine)
		assert is_prefix(old, grown)
		save(grown, args[3])
		added = {name: len(keys) - len(old.tables.get(name, [])) for name, keys in grown.tables.items() if len(keys) > len(old.tables.get(name, []))}
		print(f"wrote {args[3]}: {sum(added.values())} rows added in {len(added)} tables; vocabulary hash {grown.vocabulary_hash()}")
		for name, n in sorted(added.items(), key=lambda kv: -kv[1]):
			print(f"  {name}: +{n}")
		return 0
	print(__doc__)
	return 2


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))
