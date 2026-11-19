"""The separate critic network (config.critic_net; the user, 2026-11-18, e5's spec): its own encoders, map U-Net, tile
reader and token blocks (model_v2's trunk, one pass: it reads the recorded memory slots, never writes them), no
foresight, no second pass, no action heads. No learned weight is shared with the policy, which loses the value heads'
gradient entirely; the inference servers and the ONNX export carry the policy only (the critic never runs in games).

Its one output set, privileged (CriticHead, PrivilegedCritic's form): the change of our share's logit by day +2, +5, +7
(the change heads' scale: train.SHARE_CHANGE_CAP), the growth heads' three (G2, G5, G7: the schedule credit reads G7),
and V_end as our true share's logit as the day began plus a learned change. Every output starts at 0: no change, no
growth, V_end our share now.

The trainer (train.ppo_update) runs its trunk on every row of an update with no gradient (the per-row baselines V7, G7,
V_end and the rivals' states), trains it on one decision drawn per player-day (and the NK2 pool's rows), and takes the
link loss on those rows: MSE(policy.link_map(S_policy), S_critic.detach()), S = [global token, the mean of our heroes'
and towns' tokens, our player's token]."""
import torch
from torch import nn
from torch.nn import functional as F

from battle_model.layers import zero_output

from . import constants as c
from .model_v2 import HO, PRIVILEGED_RIVALS, PRIVILEGED_WIDTH, TO, CampaignModelV2

CRITIC_RIVAL_SLOTS = 3  # the rivals' log(Phi_rival / Phi_ours), sorted descending, 0 where none
CRITIC_EXTRA = CRITIC_RIVAL_SLOTS + 1  # and the count of rivals alive (/ PRIVILEGED_RIVALS)
CRITIC_OUT = 2 * c.N_VALUE_OFFSETS + 1  # share changes V2 V5 V7, growth G2 G5 G7, V_end's change of logit
# The trunk's modules the critic never runs (after think: the foresight, the second pass, the action and value heads)
POLICY_ONLY = (
	"foresight", "again_global", "again_hero", "again_option", "again_block", "again_norm", "move_pair", "move_mix",
	"move_mix_out", "move_mix_direct", "move_fight", "travel_hero", "travel_area", "travel_pair", "move_out",
	"cast_cell_out", "travel_out", "act_move", "act_travel", "build_identity", "build_q", "nk2_build", "nk2_upgrade",
	"nk2_trade", "trade_option", "value_heads", "growth_heads", "end_head", "privileged", "end_out", "pair_today",
	"pair_hero", "pair_tile", "pair_reach", "pair_fight", "pair_strength", "pair_rate", "link_map",
)


class CriticHead(nn.Module):
	"""The critic's outputs [B, CRITIC_OUT] from x = [our value_hidden, the rivals' value_hidden max-pooled (0 with none),
	the logit of our true share as the day began, the rivals' log Phi ratios (3, sorted descending) and their count
	alive]: a direct path from x plus one small dense layer from x, neither with a bias, both paths' outputs at 0 to
	start (PrivilegedCritic's form)."""

	def __init__(self, dim, width=PRIVILEGED_WIDTH):
		super().__init__()
		n = 2 * dim + 1 + CRITIC_EXTRA
		self.dense = nn.Linear(n, width)
		self.direct = nn.Linear(n, CRITIC_OUT, bias=False)
		self.read = nn.Linear(width, CRITIC_OUT, bias=False)
		nn.init.zeros_(self.direct.weight)
		nn.init.zeros_(self.read.weight)

	def forward(self, own, rivals, mask, share_logit, extra):
		m = mask.unsqueeze(-1) > 0
		most = torch.where(m, rivals, torch.full_like(rivals, -1e4)).amax(dim=1)
		most = torch.where(m.any(dim=1), most, torch.zeros_like(most))
		x = torch.cat([own, most, share_logit.clamp(-10.0, 10.0).unsqueeze(-1), extra], dim=-1)
		return self.direct(x) + self.read(F.gelu(self.dense(x)))


def rival_extra(phi_own, phi_rivals, mask):
	"""[B, CRITIC_EXTRA]: the rivals' log(Phi_rival / Phi_ours) sorted descending in CRITIC_RIVAL_SLOTS slots (0 where
	none), and the count of rivals alive / PRIVILEGED_RIVALS. phi_own [B], phi_rivals [B, R], mask [B, R] (1: alive with a
	state)."""
	floor = 1.0
	ratio = torch.log(phi_rivals.clamp(min=floor)) - torch.log(phi_own.clamp(min=floor)).unsqueeze(-1)
	ratio = torch.where(mask > 0, ratio, torch.full_like(ratio, -1e4))
	top = ratio.sort(dim=1, descending=True).values[:, :CRITIC_RIVAL_SLOTS]
	if top.shape[1] < CRITIC_RIVAL_SLOTS:
		top = torch.cat([top, torch.full((top.shape[0], CRITIC_RIVAL_SLOTS - top.shape[1]), -1e4, device=top.device, dtype=top.dtype)], dim=1)
	top = torch.where(top > -1e3, top.clamp(-10.0, 10.0), torch.zeros_like(top))
	count = (mask > 0).float().sum(dim=1, keepdim=True) / PRIVILEGED_RIVALS
	return torch.cat([top, count], dim=-1)


FREE_SHARE_AUX_WEIGHT = 0.25  # the free head's guess of our true share as the day began: this x a share head's mean weight


def share_prior(players):
	"""The logit of 1 / players (0 with no players given): the policy's value heads' start (model_v2.value)."""
	if players is None:
		return 0.0
	return -torch.log(players.clamp(min=2).float() - 1)


class FreeHead(nn.Module):
	"""The unprivileged head on the critic's body (the user, 2026-11-18: "add the other critics as well (for d=2,5,7)
	... 3 of every kind including the % value change critics, and probably also the auxiliary losses"): from the body's
	value_hidden alone, in the policy's old value heads' form (model_v2.value): V2 V5 V7 (share changes), G2 G5 G7 (our
	value's growth), V_end from the share prior, and one auxiliary output, our true share's logit as the day began
	(FREE_SHARE_AUX_WEIGHT); every output 0 to start. Trained on the privileged head's rows, targets and weights, its
	gradient into the body; the baselines stay the privileged head's."""

	def __init__(self, dim, share_target="share"):
		super().__init__()
		self.value_heads = zero_output(nn.Linear(dim, c.N_VALUE_OFFSETS))
		self.growth_heads = zero_output(nn.Linear(dim, c.N_VALUE_OFFSETS))
		self.end_head = zero_output(nn.Linear(dim, 1))
		self.share_head = zero_output(nn.Linear(dim, 1))
		self.share_target = share_target

	def forward(self, hidden, players=None):
		"""{"value_logit" [B, 3], "value", "growth_logit" [B, 3], "end_logit" [B], "share_now_logit" [B]}; players: the
		rows' g_players (None: a prior of 0, enough for the change heads)."""
		prior = share_prior(players)
		value_logit = self.value_heads(hidden) + (prior.unsqueeze(-1) if self.share_target == "share" and torch.is_tensor(prior) else 0.0)
		return {
			"value_logit": value_logit, "value": torch.sigmoid(value_logit), "growth_logit": self.growth_heads(hidden),
			"end_logit": self.end_head(hidden).squeeze(-1) + prior, "share_now_logit": self.share_head(hidden).squeeze(-1) + prior,
		}


class Critic(nn.Module):
	"""The critic network: trunk(x) -> (S [B, 3d], value_hidden [B, d], Phi [B]); head(...) -> raw outputs; outputs(raw,
	share_logit) -> the value heads' dict as the losses and the credit read it."""

	def __init__(self, dim=64, cell_dim=16, heads=4, layers=2, share_target="share", free_head=True):
		super().__init__()
		self.body = CampaignModelV2(dim=dim, cell_dim=cell_dim, heads=heads, layers=layers, share_target=share_target)
		for name in POLICY_ONLY:
			if hasattr(self.body, name):
				delattr(self.body, name)
		self.head = CriticHead(dim)
		self.dim = dim
		if free_head:  # (created last: the body's and the privileged head's draws as before)
			self.free = FreeHead(dim, share_target)

	def trunk(self, x):
		"""Our state at a decision: S (the link state), value_hidden, and Phi as the policy's domain knowledge computes it."""
		b = self.body
		x = b.read_inputs(x)
		known = b.domain_knowledge(x)
		state = b.encode(x)
		state = b.take_sides(state)
		state = b.join(state, known)
		state = b.see(state, known)
		state = b.advise(state, known)
		state = b.think(state)
		mine = torch.cat([x["h_present"][:, :HO], x["t_present"][:, :TO]], dim=1).float().unsqueeze(-1)
		own = torch.cat([state.heroes[:, :HO], state.towns[:, :TO]], dim=1)
		own_mean = (own * mine).sum(dim=1) / mine.sum(dim=1).clamp(min=1)
		link = torch.cat([state.glob, own_mean, state.players[:, 0]], dim=-1)
		hidden = b.value_hidden(torch.cat([link, known.phi, known.economy.to(link.dtype), known.danger.to(link.dtype)], dim=-1))
		return link, hidden, known.phi_total

	def outputs(self, raw, share_logit):
		"""{"value_logit" [B, 3], "value", "growth_logit" [B, 3], "end_logit" [B]} from the head's raw outputs."""
		k = c.N_VALUE_OFFSETS
		value_logit = raw[:, :k]
		return {"value_logit": value_logit, "value": torch.sigmoid(value_logit), "growth_logit": raw[:, k : 2 * k], "end_logit": share_logit + raw[:, 2 * k]}


def critic_kwargs(config, policy_kwargs):
	"""The critic's sizes: the policy's unless config.critic_dim / critic_cell_dim / critic_layers say."""
	return {
		"dim": int(getattr(config, "critic_dim", 0) or policy_kwargs.get("dim", 64)),
		"cell_dim": int(getattr(config, "critic_cell_dim", 0) or policy_kwargs.get("cell_dim", 16)),
		"heads": int(policy_kwargs.get("heads", 4)),
		"layers": int(getattr(config, "critic_layers", 0) or policy_kwargs.get("layers", 2)),
		"share_target": policy_kwargs.get("share_target", "share"),
	}


def critic_optimizer(critic, config, free=True):
	"""AdamW: the body at config.critic_lr (1e-4), the privileged head as the value group today (learning_rate x
	value_lr_factor), then the free head (FreeHead) at the head's rate in a third group (free=False: without it, the
	optimizer a checkpoint from before the free head saved: load_critic_optimizer)."""
	head = list(critic.head.parameters())
	free_params = list(critic.free.parameters()) if hasattr(critic, "free") else []
	ids = {id(p) for p in head + free_params}
	body = [p for p in critic.parameters() if id(p) not in ids]
	opt = torch.optim.AdamW([
		{"params": body, "name": "critic_body", "lr": float(getattr(config, "critic_lr", 1e-4)), "weight_decay": 0.0},
		{"params": head, "name": "critic_head", "lr": float(config.learning_rate * config.value_lr_factor), "weight_decay": 0.0},
	])
	if free and free_params:
		opt.add_param_group({"params": free_params, "name": "critic_free", "lr": float(config.learning_rate * config.value_lr_factor), "weight_decay": 0.0})
	return opt


def load_critic(critic, config, saved_critic, saved_optimizer):
	"""(critic, its optimizer, what was added) from a checkpoint's "critic" / "critic_optimizer": a checkpoint from before
	the free head (FreeHead) loads its body and privileged head and their Adam moments; the free head starts at 0 with
	fresh moments in its own group. Every other key must match (strict)."""
	own = critic.state_dict()
	missing = sorted(k for k in own if k not in saved_critic)
	unexpected = sorted(k for k in saved_critic if k not in own)
	if unexpected or any(not k.startswith("free.") for k in missing):
		raise RuntimeError(f"the critic's checkpoint does not fit: missing {missing[:6]}, unexpected {unexpected[:6]}")
	critic.load_state_dict({**own, **saved_critic})
	groups = len(saved_optimizer["param_groups"])
	if groups == 3 or not hasattr(critic, "free"):
		opt = critic_optimizer(critic, config)
		opt.load_state_dict(saved_optimizer)
		return critic, opt, missing
	opt = critic_optimizer(critic, config, free=False)
	opt.load_state_dict(saved_optimizer)
	opt.add_param_group({"params": list(critic.free.parameters()), "name": "critic_free", "lr": float(config.learning_rate * config.value_lr_factor), "weight_decay": 0.0})
	return critic, opt, missing
