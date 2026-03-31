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



class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device, topo_cfg=None):
        super().__init__()
        self.cfg = cfg
        self.device = device

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
        ).to(self.device)

        # Critic: ego-node embedding (hidden_dim) + drone_state (8) → value
        _hd = int(topo_cfg.hidden_dim)
        self.graph_critic_mlp = nn.Sequential(
            nn.Linear(_hd + 8, 256), nn.LayerNorm(256), nn.ELU(),
            nn.Linear(256, 128),     nn.LayerNorm(128), nn.ELU(),
            nn.Linear(128, 1),
        ).to(self.device)

        self.value_norm     = ValueNorm(1).to(self.device)
        self.gae            = GAE(0.99, 0.95)
        self.critic_loss_fn = nn.HuberLoss(delta=10)

        self.graph_optim = torch.optim.Adam(
            list(self.graph_transformer.parameters()) +
            list(self.graph_critic_mlp.parameters()),
            lr=float(self.cfg.actor.learning_rate),
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
        self.gae = GAE(0.99, 0.95)
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

    def _forward_graph(self, tensordict):
        """Graph PPO forward: topology graph → probs → node → world velocity."""
        node_features  = tensordict["agents", "observation", "node_features"]   # (B, N+1, 18)
        node_positions = tensordict["agents", "observation", "node_positions"]  # (B, N+1, 3)
        edge_features  = tensordict["agents", "observation", "edge_features"]   # (B, N+1, N+1, 7)
        node_mask      = tensordict["agents", "observation", "node_mask"].bool()  # (B, N+1)
        edge_mask      = tensordict["agents", "observation", "edge_mask"].bool()  # (B, N+1, N+1)
        drone_state    = tensordict["agents", "observation", "state"]           # (B, 8)

        # Graph transformer → probs (B, N), h (B, N+1, hidden_dim)
        probs, h = self.graph_transformer(node_features, edge_features, node_mask, edge_mask)

        # Sample action from Categorical distribution
        dist       = Categorical(probs=probs)
        action_idx = dist.sample()           # (B,)  in [0, N)
        log_prob   = dist.log_prob(action_idx)  # (B,)

        # Node index → world velocity  (pit #3: real_idx = action + 1, ego is at 0)
        N_total   = node_positions.shape[1]
        real_idx  = (action_idx + 1).clamp(max=N_total - 1)     # (B,) safety clamp
        cand_pos  = node_positions[torch.arange(node_positions.shape[0], device=self.device), real_idx]  # (B, 3)
        ego_pos   = node_positions[:, 0, :]                      # (B, 3)
        dir_vec   = cand_pos - ego_pos                           # (B, 3)
        # doc/06 §4.2.2: speed = min(v_target, dist/dt_low) so drone decelerates near node
        _dt_low   = 1.0 / 10.0                                   # high-level period = 0.1 s
        _dist     = dir_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, 1)
        _speed    = torch.minimum(
            torch.full_like(_dist, self.v_target),
            _dist / _dt_low,
        )
        vel_w     = (dir_vec / _dist) * _speed                   # (B, 3)

        # Critic value from ego embedding
        h_ego     = h[:, 0, :]                                   # (B, hidden_dim)
        value     = self.graph_critic_mlp(torch.cat([h_ego, drone_state], dim=-1))  # (B, 1)

        # Store in tensordict
        tensordict["_graph_action"]   = action_idx   # (B,) discrete node index – used in _update
        tensordict["sample_log_prob"] = log_prob      # (B,)
        tensordict["state_value"]     = value         # (B, 1)
        # Set velocity action → VelController will convert to motor thrust
        tensordict["agents", "action"] = vel_w.unsqueeze(1)  # (B, 1, 3)
        return tensordict

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _compute_values(self, tensordict):
        """Compute value estimate – handles arbitrary batch shape (B,) or (B, T)."""
        if self.graph_ppo:
            orig_shape = tensordict.batch_size          # e.g. (B,) or (B, T)
            flat_td    = tensordict.reshape(-1)          # always (B*T,)

            node_features = flat_td["agents", "observation", "node_features"]
            edge_features = flat_td["agents", "observation", "edge_features"]
            node_mask     = flat_td["agents", "observation", "node_mask"].bool()
            edge_mask     = flat_td["agents", "observation", "edge_mask"].bool()
            drone_state   = flat_td["agents", "observation", "state"]
            with torch.no_grad():
                _, h = self.graph_transformer(node_features, edge_features, node_mask, edge_mask)
            h_ego  = h[:, 0, :]                        # (B*T, hidden_dim)
            values = self.graph_critic_mlp(torch.cat([h_ego, drone_state], dim=-1))  # (B*T, 1)
            return values.view(*orig_shape, 1)         # restore batch shape
        else:
            self.feature_extractor(tensordict)
            return self.critic(tensordict)["state_value"]

    def train(self, tensordict):
        """PPO training update. Works for both cnn and graph modes."""
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            if self.graph_ppo:
                next_values = self._compute_values(next_tensordict)
            else:
                next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict)
                next_values = self.critic(next_tensordict)["state_value"]

        rewards = tensordict["next", "agents", "reward"]
        dones   = tensordict["next", "terminated"]
        values  = tensordict["state_value"].detach()
        values  = self.value_norm.denormalize(values)
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
                infos.append(self._update(minibatch))
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
        edge_features = tensordict["agents", "observation", "edge_features"]
        node_mask     = tensordict["agents", "observation", "node_mask"].bool()
        edge_mask     = tensordict["agents", "observation", "edge_mask"].bool()
        drone_state   = tensordict["agents", "observation", "state"]

        probs, h = self.graph_transformer(node_features, edge_features, node_mask, edge_mask)
        dist           = Categorical(probs=probs)
        log_probs      = dist.log_prob(tensordict["_graph_action"])  # (B,)
        action_entropy = dist.entropy()                             # (B,)

        # Critic
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

        loss = entropy_loss + actor_loss + critic_loss
        self.graph_optim.zero_grad()
        loss.backward()
        all_params = list(self.graph_transformer.parameters()) + list(self.graph_critic_mlp.parameters())
        grad_norm  = nn.utils.clip_grad.clip_grad_norm_(all_params, max_norm=5.)
        self.graph_optim.step()

        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        return TensorDict({
            "actor_loss":      actor_loss,
            "critic_loss":     critic_loss,
            "entropy":         entropy_loss,
            "actor_grad_norm": grad_norm,
            "critic_grad_norm": grad_norm,
            "explained_var":   explained_var,
        }, [])

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
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
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
