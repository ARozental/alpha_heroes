"""The credit segments' F (train.segment_shrinkage) by the types' precision-weighted spread (e5's 1b fix, 2026-11-12): with
no real gaps its mean is 1 whatever the types' sizes, so one noisy small type no longer lifts it as it lifted k7 at
v3254; a real gap in a large type lifts it; with equal sampling variances it is the old statistic."""
import numpy as np

from campaign_model.train import segment_shrinkage


def evidence(gaps):
	"""credit_evidence's {type: [n, sum, sum of squares]} from each type's per-game-player gaps."""
	return {f"T{i}": [float(len(g)), float(np.sum(g)), float(np.sum(np.square(g)))] for i, g in enumerate(gaps)}


def old_f(ev, min_players):
	"""The statistic before: the means' plain variance over their sampling variances' mean."""
	means, variances = [], []
	for n, total, squares in ev.values():
		if n < min_players:
			continue
		mean = total / n
		means.append(mean)
		variances.append(max(squares / n - mean * mean, 0.0) * n / (n - 1) / n)
	return float(np.var(means, ddof=1) / max(np.mean(variances), 1e-12))


def test_one_noisy_small_type_among_large_ones_leaves_f_near_one_on_average():
	sizes = [600] * 8 + [56]
	new, old = [], []
	for seed in range(400):
		rng = np.random.default_rng(seed)
		ev = evidence([rng.normal(0.0, 0.15 if n > 100 else 0.6, n) for n in sizes])  # (no real gaps; the small type noisier too)
		new.append(segment_shrinkage(ev, 8)[1])
		old.append(old_f(ev, 8))
	assert abs(np.mean(new) - 1.0) < 0.12, np.mean(new)
	assert np.mean(np.array(old) > 2.0) > 3 * np.mean(np.array(new) > 2.0)  # (the old one far more often well above 1 on noise alone)


def test_a_real_gap_in_a_large_type_lifts_it():
	rng = np.random.default_rng(1)
	gaps = [rng.normal(0.0, 0.15, 600) for _ in range(8)] + [rng.normal(0.0, 0.6, 56)]
	gaps[0] = gaps[0] + 0.03  # (a real gap: ~5 sampling sd of a type of 600)
	k, f, types = segment_shrinkage(evidence(gaps), 8)
	assert types == 9 and f > 2.5 and k > 0.6


def test_with_equal_sampling_variances_it_is_the_old_statistic():
	rng = np.random.default_rng(2)
	base = rng.normal(0.0, 1.0, 50)
	gaps = [(base - base.mean()) / base.std(ddof=1) * 0.2 + shift for shift in (0.0, 0.01, -0.02, 0.05)]  # (the same spread in every type)
	ev = evidence(gaps)
	assert abs(segment_shrinkage(ev, 8)[1] - old_f(ev, 8)) < 1e-9
