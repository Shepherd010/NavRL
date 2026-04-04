import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world


def _resolve_amp_dtype(device: str) -> torch.dtype | None:
    """Return BF16 if supported, FP16 as fallback, None if CUDA unavailable."""
    if not torch.cuda.is_available():
        return None
    # BF16: safer for RL (no loss of dynamic range vs FP16's narrow range)
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device, topo_cfg=None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        # Optional frozen-teacher distillation module (off by default).
        self.teacher_distill_enabled = False
        self.teacher_policy = None
        self.teacher_distill_weight = 0.0
        self.teacher_distill_min_weight = 0.0
        self.teacher_distill_warmup_steps = 0
        self.teacher_distill_decay_steps = 0
        self.ppo_prior_enabled = False
        self.ppo_prior_policy = None
        self.ppo_prior_weight = 0.0
        self.ppo_prior_min_weight = 0.0
        self.ppo_prior_warmup_steps = 0
        self.ppo_prior_decay_steps = 0
        self.ppo_prior_temperature = 1.0
        self._graph_update_steps = 0

        # Detect graph_ppo mode from observation spec
        _obs_keys = set(observation_spec["agents"]["observation"].keys())
        self.graph_ppo = ("node_features" in _obs_keys) and (topo_cfg is not None)

        if self.graph_ppo:
            self._build_graph_ppo(topo_cfg)
        else:
            self._build_cnn_ppo(cfg, observation_spec, action_spec)

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_graph_ppo(self, topo_cfg):
        """Build GraphTransformer-based policy + critic."""
        sys.path.insert(0, os.path.dirname(__file__))
        from graph_transformer import GraphTransformer

        self.topo_cfg = topo_cfg
        self.v_target = float(topo_cfg.v_target)
        self.min_dist = float(topo_cfg.min_dist)
        # discrete action entropy has different scale from continuous Beta entropy
        self.entropy_coeff = float(topo_cfg.entropy_loss_coefficient)

        self.graph_transformer = GraphTransformer(
            node_feat_dim=int(topo_cfg.node_feat_dim),
            edge_feat_dim=int(topo_cfg.edge_feat_dim),
            hidden_dim=int(topo_cfg.hidden_dim),
            num_heads=int(topo_cfg.num_heads),
            num_layers=int(topo_cfg.num_layers),
            dropout=float(topo_cfg.dropout),
            use_spd_bias=bool(getattr(topo_cfg, 'use_spd_bias', False)),
            use_topo_bias=bool(getattr(topo_cfg, 'use_topo_bias', False)),
            sparse_topk=int(getattr(topo_cfg, 'sparse_topk', 8)),
        ).to(self.device)

        # Critic: ego-node embedding (hidden_dim) + drone_state (8) → value
        _hd = int(topo_cfg.hidden_dim)
        self.graph_critic_mlp = nn.Sequential(
            nn.Linear(_hd + 8, 256), nn.LayerNorm(256), nn.ELU(),
            nn.Linear(256, 128),     nn.LayerNorm(128), nn.ELU(),
            nn.Linear(128, 1),
        ).to(self.device)

        self.value_norm     = ValueNorm(1).to(self.device)
        _gamma = float(getattr(self.cfg, 'gamma', 0.995))
        self.gae            = GAE(_gamma, 0.95)
        self.critic_loss_fn = nn.HuberLoss(delta=10)

        # AMP setup: BF16 preferred, FP16 fallback, disabled if no CUDA.
        # ValueNorm, GAE, and HuberLoss stay in FP32 (handled outside autocast).
        self._amp_dtype = _resolve_amp_dtype(self.device)
        self._amp_enabled = self._amp_dtype is not None
        # GradScaler only needed for FP16 (BF16 has no underflow problem)
        self._scaler = (
            torch.cuda.amp.GradScaler()
            if (self._amp_enabled and self._amp_dtype == torch.float16)
            else None
        )

        # Separate optimisers for actor (transformer) and critic (mlp) so that
        # their learning rates can be tuned independently (key for stable PPO).
        self.graph_actor_optim = torch.optim.Adam(
            self.graph_transformer.parameters(),
            lr=float(self.cfg.actor.learning_rate),
        )
        self.graph_critic_optim = torch.optim.Adam(
            self.graph_critic_mlp.parameters(),
            lr=float(self.cfg.critic.learning_rate),
        )

    def _build_cnn_ppo(self, cfg, observation_spec, action_spec):
        """Build original CNN-based PPO (unchanged from original)."""
        # Feature extractor for LiDAR
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)
        
        # Dynamic obstacle information extractor
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)

        # Feature extractor
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        # Actor network
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        _gamma = float(getattr(cfg, 'gamma', 0.995))
        self.gae = GAE(_gamma, 0.95)
        self.critic_loss_fn = nn.HuberLoss(delta=10)

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.actor.learning_rate)

        # Dummy Input for nn lazymodule initialization
        dummy_input = observation_spec.zero()
        self._forward_ppo(dummy_input)

        # Initialize network weights
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)

    # ------------------------------------------------------------------
    # Optional Teacher Distillation (graph_ppo)
    # ------------------------------------------------------------------

    def enable_teacher_distill(self, teacher_policy, distill_cfg):
        """Attach a frozen CNN-PPO teacher for graph_ppo velocity distillation."""
        from omegaconf import OmegaConf, DictConfig
        self.teacher_policy = teacher_policy
        self.teacher_distill_enabled = True
        # Use OmegaConf.select for DictConfig (getattr raises ConfigAttributeError on missing keys)
        _get = (lambda c, k, d: OmegaConf.select(c, k, default=d)) if isinstance(distill_cfg, DictConfig) \
               else (lambda c, k, d: getattr(c, k, d))
        self.teacher_distill_weight = float(_get(distill_cfg, 'weight', 1.0))
        self.teacher_distill_min_weight = float(_get(distill_cfg, 'min_weight', 0.0))
        self.teacher_distill_warmup_steps = int(_get(distill_cfg, 'warmup_steps', 0))
        self.teacher_distill_decay_steps = int(_get(distill_cfg, 'decay_steps', 0))

    def disable_teacher_distill(self):
        """Detach teacher distillation (curriculum stage switch)."""
        self.teacher_distill_enabled = False
        self.teacher_policy = None

    def enable_ppo_prior(self, prior_policy, prior_cfg):
        """Attach a frozen CNN-PPO prior to warm-start Graph-PPO action selection."""
        from omegaconf import OmegaConf, DictConfig
        self.ppo_prior_policy = prior_policy
        self.ppo_prior_enabled = True
        _get = (lambda c, k, d: OmegaConf.select(c, k, default=d)) if isinstance(prior_cfg, DictConfig) \
               else (lambda c, k, d: getattr(c, k, d))
        self.ppo_prior_weight = float(_get(prior_cfg, 'weight', 1.0))
        self.ppo_prior_min_weight = float(_get(prior_cfg, 'min_weight', 0.0))
        self.ppo_prior_warmup_steps = int(_get(prior_cfg, 'warmup_steps', 0))
        self.ppo_prior_decay_steps = int(_get(prior_cfg, 'decay_steps', 0))
        self.ppo_prior_temperature = float(_get(prior_cfg, 'temperature', 1.0))

    def disable_ppo_prior(self):
        self.ppo_prior_enabled = False
        self.ppo_prior_policy = None

    def _ppo_prior_weight_now(self):
        if not self.ppo_prior_enabled:
            return 0.0

        base_w = self.ppo_prior_weight
        min_w = self.ppo_prior_min_weight
        step = self._graph_update_steps
        warmup = self.ppo_prior_warmup_steps
        decay = self.ppo_prior_decay_steps

        if step < warmup:
            return base_w
        if decay <= 0:
            return min_w

        p = min(1.0, float(step - warmup) / float(decay))
        return base_w + (min_w - base_w) * p

    def _teacher_distill_weight_now(self):
        if not self.teacher_distill_enabled:
            return 0.0

        base_w = self.teacher_distill_weight
        min_w = self.teacher_distill_min_weight
        step = self._graph_update_steps
        warmup = self.teacher_distill_warmup_steps
        decay = self.teacher_distill_decay_steps

        if step < warmup:
            return base_w
        if decay <= 0:
            return min_w

        p = min(1.0, float(step - warmup) / float(decay))
        return base_w + (min_w - base_w) * p

    @torch.no_grad()
    def _cnn_velocity_from_obs(self, cnn_policy, tensordict):
        """Run frozen CNN PPO policy on current obs and return world-frame velocity (B, 3).

        Uses the deterministic Beta-distribution mean as action (alpha/(alpha+beta)),
        which is identical to the _forward_ppo path minus the stochastic sample.
        Keeps shapes consistent with how _forward_ppo maps action_normalized → world vel.
        """
        if cnn_policy is None:
            raise RuntimeError("CNN prior/teacher policy is None")

        # Build a minimal tensordict with only the CNN-PPO obs keys.
        # batch_size must match the minibatch (first dim of tensordict).
        B = tensordict.batch_size[0]
        td_teacher = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "lidar":            tensordict["agents", "observation", "lidar"],
                    "dynamic_obstacle": tensordict["agents", "observation", "dynamic_obstacle"],
                    "state":            tensordict["agents", "observation", "state"],
                    "direction":        tensordict["agents", "observation", "direction"],
                }, batch_size=[B]),
            }, batch_size=[B]),
        }, batch_size=[B], device=self.device)

        # Run feature extractor (writes "_feature" into td_teacher).
        cnn_policy.feature_extractor(td_teacher)

        # Run the BetaActor head directly to get alpha/beta without TorchRL sampling.
        # teacher.actor is a ProbabilisticActor wrapping TensorDictModule(BetaActor).
        # Calling it directly would sample from Beta; we want the mean instead.
        # Access the inner TensorDictModule (first element in the sequential) and call it.
        # actor.module is nn.ModuleList([TensorDictModule(BetaActor), ...]).
        # Index [0] is the TensorDictModule that reads "_feature" and writes alpha/beta.
        cnn_policy.actor.module[0](td_teacher)     # writes alpha, beta → td_teacher

        alpha = td_teacher["alpha"]   # (B, action_dim) or (B, 1, action_dim)
        beta  = td_teacher["beta"]

        # Deterministic mean of Beta distribution: μ = alpha / (alpha + beta)
        # Parentheses are critical: must divide alpha by (alpha+beta), not (alpha+beta-clamp).
        action_norm = alpha / (alpha + beta + 1e-6)          # (B, action_dim) or (B, 1, action_dim)

        # Map [0,1] → [-limit, +limit], same formula as _forward_ppo.
        limit = cnn_policy.cfg.actor.action_limit
        action_local = (2.0 * action_norm * limit) - limit   # (B, ..., 3)

        # vec_to_world expects (B, 3) or (B, 1, 3); direction is (B, 1, 3).
        direction = td_teacher["agents", "observation", "direction"]
        action_world = vec_to_world(action_local, direction)  # (B, 1, 3) or (B, 3)

        # Normalise to (B, 3) so distillation MSE is straightforward.
        if action_world.dim() == 3:
            action_world = action_world.squeeze(1)            # (B, 3)
        return action_world

    @torch.no_grad()
    def _teacher_velocity_from_obs(self, tensordict):
        if self.teacher_policy is None:
            raise RuntimeError("Teacher distillation is enabled but teacher_policy is None")
        return self._cnn_velocity_from_obs(self.teacher_policy, tensordict)

    def _candidate_node_velocities(self, node_positions: torch.Tensor) -> torch.Tensor:
        """Map graph candidate nodes to world-frame target velocities. Returns (B, N, 3)."""
        ego_pos = node_positions[:, :1, :]               # (B, 1, 3)
        cand_pos = node_positions[:, 1:, :]              # (B, N, 3)
        dir_vec = cand_pos - ego_pos                     # (B, N, 3)
        node_dist = dir_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dt_low = 1.0 / float(getattr(self.topo_cfg, 'low_freq', 10.0))
        speed = torch.minimum(
            torch.full_like(node_dist, self.v_target),
            node_dist / dt_low,
        )
        cand_vel = (dir_vec / node_dist) * speed
        cand_vel = cand_vel.clone()
        cand_vel[..., 2] = 0.0
        return cand_vel

    def _apply_ppo_prior(self, probs, node_positions, node_mask, tensordict):
        """Blend graph action distribution with PPO prior projected onto graph nodes."""
        prior_weight = self._ppo_prior_weight_now()
        if (not self.ppo_prior_enabled) or self.ppo_prior_policy is None or prior_weight <= 0.0:
            return probs, 0.0

        with torch.no_grad():
            prior_vel = self._cnn_velocity_from_obs(self.ppo_prior_policy, tensordict).to(probs.dtype)

        cand_vel = self._candidate_node_velocities(node_positions).to(probs.dtype)
        cand_mask = node_mask[:, 1:].bool()
        temp = max(float(self.ppo_prior_temperature), 1e-6)
        sq_err = ((cand_vel - prior_vel.unsqueeze(1)) ** 2).sum(dim=-1)
        prior_logits = -sq_err / temp
        prior_logits = prior_logits.masked_fill(~cand_mask, -1e9)
        prior_probs = F.softmax(prior_logits, dim=-1)

        probs = (1.0 - prior_weight) * probs + prior_weight * prior_probs
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return probs, prior_weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, tensordict):
        if self.graph_ppo:
            return self._forward_graph(tensordict)
        return self._forward_ppo(tensordict)

    def _forward_ppo(self, tensordict):
        """Original CNN PPO forward (unchanged)."""
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        # Coordinate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

    @torch.no_grad()
    def _forward_graph(self, tensordict):
        """Graph PPO forward: topology graph → probs → node → world velocity."""
        # @torch.no_grad(): rollout collection only needs sampled actions/values as data;
        # _update_graph does a fresh differentiable forward pass during training.
        node_features  = tensordict["agents", "observation", "node_features"]   # (B, N+1, 18)
        node_positions = tensordict["agents", "observation", "node_positions"]  # (B, N+1, 3)
        edge_features  = tensordict["agents", "observation", "edge_features"]   # (B, N+1, N+1, 8)
        node_mask      = tensordict["agents", "observation", "node_mask"].bool()  # (B, N+1)
        edge_mask      = tensordict["agents", "observation", "edge_mask"].bool()  # (B, N+1, N+1)
        drone_state    = tensordict["agents", "observation", "state"]           # (B, 8)

        # Precompute SPD matrix only when model actually uses SPD bias.
        spd_matrix = None
        if self.graph_transformer.need_spd_matrix():
            spd_matrix = self.graph_transformer._compute_spd_matrix(edge_mask)  # (B, N+1, N+1)
            tensordict["_spd_matrix"] = spd_matrix

        # Graph transformer → probs (B, N), h (B, N+1, hidden_dim)
        # Run under AMP autocast for 2-3× Transformer speedup.
        with torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self._amp_dtype):
            probs, h = self.graph_transformer(node_features, edge_features, node_mask, edge_mask,
                                              spd_matrix=spd_matrix)
        # Upcast to FP32 before Categorical / arithmetic (stability)
        probs = probs.float()
        h     = h.float()
        probs, ppo_prior_weight = self._apply_ppo_prior(probs, node_positions, node_mask, tensordict)

        # Sample action from Categorical distribution
        dist       = Categorical(probs=probs)
        action_idx = dist.sample()           # (B,)  in [0, N)
        log_prob   = dist.log_prob(action_idx)  # (B,)

        # Node index → world velocity  (pit #3: real_idx = action + 1, ego is at 0)
        N_total   = node_positions.shape[1]
        real_idx  = (action_idx + 1).clamp(max=N_total - 1)     # (B,) safety clamp
        cand_pos  = node_positions[torch.arange(node_positions.shape[0], device=self.device), real_idx]  # (B, 3)
        cand_vel_all = self._candidate_node_velocities(node_positions)
        vel_w = cand_vel_all[torch.arange(node_positions.shape[0], device=self.device), action_idx]

        # Critic value from ego embedding
        h_ego     = h[:, 0, :]                                   # (B, hidden_dim)
        value     = self.graph_critic_mlp(torch.cat([h_ego, drone_state], dim=-1))  # (B, 1)

        # Store in tensordict
        tensordict["graph_action"]    = action_idx   # (B,) discrete node index – used in _update
        tensordict["sample_log_prob"] = log_prob      # (B,)
        tensordict["state_value"]     = value         # (B, 1)
        tensordict["_ppo_prior_weight"] = torch.full((node_positions.shape[0],), ppo_prior_weight, device=self.device)
        # Cache raw velocity so _pre_sim_step_graph can read it AFTER VelController runs.
        # VelController._inv_call overwrites ("agents","action") with motor thrusts before
        # _pre_sim_step is called, so reading from ("agents","action") inside _pre_sim_step
        # would give motor thrusts, not the velocity we want for the QP safety filter.
        tensordict["_v_rl"] = vel_w                   # (B, 3) raw policy velocity
        # Set velocity: VelController expects (B, 3) and unsqueezes internally to (B, 1, 3)
        tensordict["agents", "action"] = vel_w        # (B, 3)
        return tensordict

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _compute_values(self, tensordict):
        """Compute value estimate – handles arbitrary batch shape (B,) or (B, T).
        Graph path processes in chunks to avoid O(N²) edge memory explosion.
        """
        if self.graph_ppo:
            orig_shape = tensordict.batch_size          # e.g. (B,) or (B, T)
            flat_td    = tensordict.reshape(-1)          # always (B*T,)
            total      = flat_td.batch_size[0]
            chunk_size = int(getattr(self.topo_cfg, 'value_chunk_size', 256))

            node_features = flat_td["agents", "observation", "node_features"]
            edge_features = flat_td["agents", "observation", "edge_features"]
            node_mask     = flat_td["agents", "observation", "node_mask"].bool()
            edge_mask     = flat_td["agents", "observation", "edge_mask"].bool()
            drone_state   = flat_td["agents", "observation", "state"]

            chunks = []
            with torch.no_grad():
                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    # Reuse cached SPD matrix if present — avoids redundant Floyd-Warshall.
                    # The flat slice preserves the same index range as the observation slices.
                    spd_chunk = None
                    if "_spd_matrix" in flat_td.keys():
                        spd_chunk = flat_td["_spd_matrix"][start:end]
                    with torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self._amp_dtype):
                        _, h = self.graph_transformer(
                            node_features[start:end], edge_features[start:end],
                            node_mask[start:end],     edge_mask[start:end],
                            spd_matrix=spd_chunk,
                        )
                    h_ego = h[:, 0, :].float()              # (chunk, hidden_dim) – FP32
                    v = self.graph_critic_mlp(torch.cat([h_ego, drone_state[start:end]], dim=-1))
                    chunks.append(v)
            values = torch.cat(chunks, dim=0)       # (B*T, 1)
            return values.view(*orig_shape, 1)      # restore batch shape
        else:
            self.feature_extractor(tensordict)
            return self.critic(tensordict)["state_value"]

    def train(self, mode_or_tensordict=True):
        """Support both nn.Module.train(mode) and PPO train(update_td).

        PyTorch calls ``self.train(False)`` from ``eval()``. This class also exposes
        the historical ``policy.train(tensordict)`` API used by the training loop.
        Dispatch on argument type so both semantics remain valid.
        """
        if isinstance(mode_or_tensordict, bool):
            return super().train(mode_or_tensordict)

        tensordict = mode_or_tensordict
        super().train(True)

        # PPO training update path. Works for both cnn and graph modes.
        next_tensordict = tensordict["next"]

        rewards = tensordict["next", "agents", "reward"]
        dones   = tensordict["next", "terminated"]
        values  = tensordict["state_value"].detach()
        values  = self.value_norm.denormalize(values)

        if self.graph_ppo:
            # Shift trick: for t=0..T-2, next_state == state at t+1, so V(s_{t+1})
            # is already in `values[:,1:]` — no extra forward pass needed.
            # We only need to compute the bootstrap value V(s_T) for the LAST step.
            # This reduces _compute_values from 128 chunks → 4 chunks (32× speedup).
            # Safety: when done=True, GAE multiplies next_value by 0, so any value is fine.
            next_values = torch.empty_like(values)          # (B, T, 1)
            next_values[:, :-1] = values[:, 1:]             # shift: already denormalized
            last_boot = self._compute_values(next_tensordict[:, -1:])  # (B, 1, 1)
            next_values[:, -1:] = self.value_norm.denormalize(last_boot)
        else:
            with torch.no_grad():
                next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict)
                next_values = self.critic(next_tensordict)["state_value"]
            next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv = (adv - adv.mean()) / adv.std().clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)
        # detach before storing: adv/ret are constants (baselines) in PPO update
        tensordict.set("adv", adv.detach())
        tensordict.set("ret", ret.detach())

        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch).to('cpu'))
        infos = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict):
        """Minibatch PPO update. Dispatches to graph or cnn path."""
        if self.graph_ppo:
            return self._update_graph(tensordict)
        return self._update_cnn(tensordict)

    def _update_graph(self, tensordict):
        """Graph PPO minibatch update (Categorical action)."""
        node_features = tensordict["agents", "observation", "node_features"]
        node_positions = tensordict["agents", "observation", "node_positions"]
        edge_features = tensordict["agents", "observation", "edge_features"]
        node_mask     = tensordict["agents", "observation", "node_mask"].bool()
        edge_mask     = tensordict["agents", "observation", "edge_mask"].bool()
        drone_state   = tensordict["agents", "observation", "state"]

        # Reuse SPD matrix cached during rollout — avoids Floyd-Warshall on every minibatch
        spd_matrix = tensordict.get("_spd_matrix", None)

        with torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self._amp_dtype):
            probs, h = self.graph_transformer(node_features, edge_features, node_mask, edge_mask,
                                              spd_matrix=spd_matrix)
        # Upcast to FP32 before loss computation (Categorical, log_prob, PPO ratio)
        probs = probs.float()
        h     = h.float()
        probs, ppo_prior_weight = self._apply_ppo_prior(probs, node_positions, node_mask, tensordict)
        dist           = Categorical(probs=probs)
        log_probs      = dist.log_prob(tensordict["graph_action"])   # (B,)
        action_entropy = dist.entropy()                             # (B,)

        # Critic (FP32 headcritical for numerical stability in value loss)
        h_ego  = h[:, 0, :]
        value  = self.graph_critic_mlp(torch.cat([h_ego, drone_state], dim=-1))  # (B, 1)

        # Entropy loss — use graph-specific coefficient (discrete entropy scale differs)
        entropy_loss = -self.entropy_coeff * action_entropy.mean()

        # Actor PPO loss (clipped surrogate)
        # detach old log_probs and advantage — they are constants from rollout collection
        advantage    = tensordict["adv"].detach()
        old_log_prob = tensordict["sample_log_prob"].detach()
        ratio  = torch.exp(log_probs - old_log_prob).unsqueeze(-1)
        surr1  = advantage * ratio
        surr2  = advantage * ratio.clamp(1. - self.cfg.actor.clip_ratio, 1. + self.cfg.actor.clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2))

        # Critic loss (b_value is the old estimate — treated as constant)
        b_value       = tensordict["state_value"].detach()
        ret           = tensordict["ret"]
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio)
        critic_loss  = torch.max(self.critic_loss_fn(ret, value_clipped), self.critic_loss_fn(ret, value))

        distill_loss = torch.zeros([], device=self.device)
        distill_weight = 0.0
        if self.teacher_distill_enabled and self.teacher_policy is not None:
            # Student expected velocity from graph action probabilities.
            # Candidate nodes exclude ego node at index 0.
            cand_vel = self._candidate_node_velocities(node_positions)
            v_student = (probs.unsqueeze(-1) * cand_vel).sum(dim=1)  # (B, 3)

            # Frozen teacher target velocity from the same minibatch observations.
            v_teacher = self._teacher_velocity_from_obs(tensordict).to(v_student.dtype)

            distill_loss = F.mse_loss(v_student, v_teacher)
            distill_weight = self._teacher_distill_weight_now()

        loss = entropy_loss + actor_loss + critic_loss + (distill_weight * distill_loss)
        self.graph_actor_optim.zero_grad(set_to_none=True)
        self.graph_critic_optim.zero_grad(set_to_none=True)
        if self._scaler is not None:
            # FP16 path: scale loss, unscale each optimizer separately, then clip & step
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.graph_actor_optim)
            self._scaler.unscale_(self.graph_critic_optim)
            actor_grad_norm  = nn.utils.clip_grad.clip_grad_norm_(self.graph_transformer.parameters(), max_norm=5.)
            critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.graph_critic_mlp.parameters(), max_norm=5.)
            self._scaler.step(self.graph_actor_optim)
            self._scaler.step(self.graph_critic_optim)
            self._scaler.update()
        else:
            # BF16 or FP32: no scaler needed
            loss.backward()
            actor_grad_norm  = nn.utils.clip_grad.clip_grad_norm_(self.graph_transformer.parameters(), max_norm=5.)
            critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.graph_critic_mlp.parameters(), max_norm=5.)
            self.graph_actor_optim.step()
            self.graph_critic_optim.step()

        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        self._graph_update_steps += 1
        return TensorDict({
            "actor_loss":      actor_loss,
            "critic_loss":     critic_loss,
            "entropy":         entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var":   explained_var,
            "distill_loss":    distill_loss,
            "distill_weight":  torch.tensor(distill_weight, device=self.device),
            "ppo_prior_weight": torch.tensor(ppo_prior_weight, device=self.device),
        }, [])

    @torch.no_grad()
    def get_viz_data(self, tensordict):
        """Return intermediate quantities needed for visualization (V1, V2, V3).
        Operates on a single-frame tensordict, e.g. data[:, -1].
        Returns a plain dict of CPU tensors — safe to pass directly to viz.py functions.
        """
        if not self.graph_ppo:
            return {}

        node_features  = tensordict["agents", "observation", "node_features"][:1]  # env 0 only
        node_positions = tensordict["agents", "observation", "node_positions"][:1]
        edge_features  = tensordict["agents", "observation", "edge_features"][:1]
        node_mask      = tensordict["agents", "observation", "node_mask"][:1].bool()
        edge_mask      = tensordict["agents", "observation", "edge_mask"][:1].bool()
        drone_state    = tensordict["agents", "observation", "state"][:1]

        with torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self._amp_dtype):
            probs, h, all_attn = self.graph_transformer.forward_with_attn(
                node_features, edge_features, node_mask, edge_mask
            )
        probs = probs.float()
        h     = h.float()

        h_ego  = h[:, 0, :]
        value  = self.graph_critic_mlp(torch.cat([h_ego, drone_state], dim=-1))

        action_idx = tensordict.get("graph_action", None)
        selected   = action_idx[:1].cpu() if action_idx is not None else None

        # Safe optional fetch — avoids Python 'and' on a Tensor (ambiguous boolean)
        v_rl_key  = "_v_rl"
        v_rl_val  = tensordict[v_rl_key][:1].cpu() if v_rl_key in tensordict.keys() else None

        return {
            "node_positions": node_positions[0].cpu(),   # (N+1, 3)
            "node_mask":      node_mask[0].cpu(),        # (N+1,)  bool
            "edge_mask":      edge_mask[0].cpu(),        # (N+1, N+1) bool
            "probs":          probs[0].cpu(),            # (N,)
            "node_features":  node_features[0].cpu(),   # (N+1, node_feat_dim)
            "all_attn":       all_attn,                  # list[L] of (1, H, N+1, N+1) CPU
            "value":          value[0].cpu(),            # (1,)
            "selected_idx":   selected,                  # (1,) or None
            "v_rl":           v_rl_val,
        }

    def _update_cnn(self, tensordict):
        """Original CNN PPO minibatch update (unchanged)."""
        self.feature_extractor(tensordict)

        action_dist    = self.actor.get_dist(tensordict)
        log_probs      = action_dist.log_prob(tensordict[("agents", "action_normalized")])
        action_entropy = action_dist.entropy()
        entropy_loss   = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        advantage = tensordict["adv"]
        ratio     = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1     = advantage * ratio
        surr2     = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

        b_value      = tensordict["state_value"]
        ret          = tensordict["ret"]
        value        = self.critic(tensordict)["state_value"] 
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio)
        critic_loss  = torch.max(self.critic_loss_fn(ret, value_clipped), self.critic_loss_fn(ret, value))

        loss = entropy_loss + actor_loss + critic_loss
        self.feature_extractor_optim.zero_grad(set_to_none=True)
        self.actor_optim.zero_grad(set_to_none=True)
        self.critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        actor_grad_norm  = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()

        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        return TensorDict({
            "actor_loss":      actor_loss,
            "critic_loss":     critic_loss,
            "entropy":         entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var":   explained_var,
        }, [])
