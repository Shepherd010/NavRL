"""
TopoExtractor: LiDAR -> Voronoi topology graph

Pure PyTorch, no Isaac Sim dependency. All heavy computation is batched on GPU.

Input:
    ray_hits_w  : (B, N_rays, 3)   LiDAR ray hit points in world frame
    ray_pos_w   : (B, 3)           LiDAR sensor world position (= ego position)
    dyn_obs     : (B, N_dyn, 10)   dynamic obstacle states (already squeezed, no dim-1)
    ego_vel     : (B, 3)           ego velocity (world frame)
    target_pos  : (B, 3)           goal position (world frame)

Output dict:
    node_features  : (B, max_nodes+1, 18)
    node_positions : (B, max_nodes+1, 3)   world-frame; index 0 = ego
    edge_features  : (B, max_nodes+1, max_nodes+1, 8)
    node_mask      : (B, max_nodes+1)  bool, True = valid
    edge_mask      : (B, max_nodes+1, max_nodes+1)  bool
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TopoExtractor:

    def __init__(self, cfg):
        self.max_nodes   = cfg.max_nodes          # 50, not counting ego
        self.safe_radius = cfg.safe_radius         # 0.4 m
        self.grad_threshold = cfg.grad_threshold   # 0.5
        self.node_feat_dim = cfg.node_feat_dim     # 18
        self.edge_feat_dim = cfg.edge_feat_dim     # 8 (rel_pos×3 + elen×1 + dyn_norm×1 + dir×3)
        self.lidar_range = cfg.lidar_range         # 10.0
        self.grid_size   = cfg.grid_size           # 100
        self.nms_radius  = cfg.nms_radius          # 0.5
        self.k_neighbors = cfg.k_neighbors         # 5
        self.max_edge_length = cfg.max_edge_length # 2.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_topology(
        self,
        ray_hits_w: torch.Tensor,   # (B, N_rays, 3)
        ray_pos_w:  torch.Tensor,   # (B, 3)
        dyn_obs:    torch.Tensor,   # (B, N_dyn, 10)
        ego_vel:    torch.Tensor,   # (B, 3)
        target_pos: torch.Tensor,   # (B, 3)
    ) -> Dict[str, torch.Tensor]:

        B      = ray_hits_w.shape[0]
        device = ray_hits_w.device

        # Sanitize NaN in LiDAR
        ray_hits_w = torch.nan_to_num(ray_hits_w, nan=0.0)

        # ---------- 1. 2D distance field ----------
        dist_field, gradient, grid_xy = self._compute_distance_field(ray_hits_w, ray_pos_w)
        # dist_field: (B, H, W), gradient: (B, H, W, 2), grid_xy: (H, W, 2) local ego-frame

        # ---------- 2. Voronoi nodes (local ego-frame) ----------
        nodes_local, clearances, nodes_mask = self._find_voronoi_nodes(
            dist_field, gradient, grid_xy,
            ego_pos=ray_pos_w, target_pos=target_pos,
        )
        # nodes_local : (B, max_nodes, 3)  xy local + z=0
        # nodes_mask  : (B, max_nodes)     bool

        # ---------- 3. Convert to world frame, prepend ego as index-0 ----------
        ego_pos = ray_pos_w  # (B, 3)

        nodes_world = nodes_local.clone()
        nodes_world[..., :2] += ego_pos[:, :2].unsqueeze(1)   # add x,y offset
        nodes_world[..., 2]   = ego_pos[:, 2:3].expand(-1, self.max_nodes)  # z = ego height

        # Prepend ego node: shape (B, max_nodes+1, 3)
        ego_node = ego_pos.unsqueeze(1)              # (B, 1, 3)
        node_positions = torch.cat([ego_node, nodes_world], dim=1)  # (B, N+1, 3)

        # Full node mask: ego is always valid
        ego_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
        full_node_mask = torch.cat([ego_valid, nodes_mask], dim=1)  # (B, N+1)

        # ---------- Empty-graph fallback (spec §3.1) ----------
        # If an environment has zero candidate nodes after voronoi extraction, place
        # one virtual node in the target direction so the action space is never empty
        # and softmax never receives all -1e9 logits (→ NaN).
        empty_env = ~nodes_mask.any(dim=1)  # (B,) bool: True where no candidate found
        if empty_env.any():
            logger.debug("Empty-graph fallback triggered for %d/%d envs", empty_env.sum().item(), B)
            tgt_dir   = target_pos - ego_pos                 # (B, 3)
            tgt_dir_n = F.normalize(tgt_dir, dim=-1)         # unit vector (B, 3)
            virtual   = ego_pos + tgt_dir_n                  # 1 m ahead in world frame
            # Write into slot index 1 (first candidate slot) for empty envs
            node_positions[empty_env, 1]  = virtual[empty_env]
            full_node_mask[empty_env, 1]  = True
            # clearances stays 0 for virtual node (already zero-padded)

        # ---------- Goal-direction waypoint — always present (last candidate slot) ----------
        # Regardless of Voronoi nodes found, always overwrite the LAST candidate slot with a
        # waypoint in the goal direction.  This guarantees the policy always has a direct
        # "fly toward goal" option to learn from during exploration, which is critical in
        # early training when policy weights are random and nodes may be clustered sideways.
        # The QP safety shield will deflect the resulting velocity if the path is blocked,
        # so injecting this node does NOT bypass obstacle avoidance.
        #
        # CRITICAL: look_ahead MUST be < max_edge_length (2.0 m) so that _build_edges
        # creates a real edge ego→goal_wp.  Without an edge the sparse attention layer
        # cannot route ego information to/from goal_wp, making the node effectively
        # invisible → policy can never learn to select it reliably.
        if target_pos is not None:
            tgt_dir    = target_pos - ego_pos                              # (B, 3)
            tgt_dist   = tgt_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, 1)
            tgt_dir_n  = tgt_dir / tgt_dist                                # unit (B, 3)
            # Look-ahead: 1.8 m (just under max_edge_length=2.0 m to guarantee an edge),
            # or the actual goal distance when closer than 1.8 m.
            look_ahead = tgt_dist.squeeze(-1).clamp(max=1.8)              # (B,)
            goal_wp    = ego_pos.clone()
            goal_wp[:, :2] += tgt_dir_n[:, :2] * look_ahead.unsqueeze(-1)
            goal_wp[:, 2]   = ego_pos[:, 2]     # keep ego height
            node_positions[:, -1] = goal_wp     # last slot → always reachable as action N-1
            full_node_mask[:, -1] = True

        # ---------- 4. Edges ----------
        edges, edge_lengths = self._build_edges(node_positions, full_node_mask)
        # edges: (B, N+1, N+1) bool

        # Force edges between ego (index 0) and the goal waypoint (index -1) so
        # that sparse attention can propagate ego context to goal_wp even when
        # _build_edges would otherwise miss it (e.g. KNN of 5 already taken by
        # closer Voronoi nodes).  Symmetrized: both directions added.
        if target_pos is not None:
            edges[:, 0, -1] = True
            edges[:, -1, 0] = True
            # Update edge_lengths for the newly forced edges
            d_ego_goalwp = (node_positions[:, -1] - node_positions[:, 0]).norm(dim=-1)  # (B,)
            edge_lengths[:, 0, -1] = d_ego_goalwp
            edge_lengths[:, -1, 0] = d_ego_goalwp

        # ---------- 5. Node features ----------
        node_features = self._encode_node_features(
            node_positions, clearances, full_node_mask,
            ego_pos, ego_vel, target_pos
        )  # (B, N+1, 18)

        # ---------- 6. Edge features ----------
        edge_features = self._encode_edge_features(
            node_positions, edges, edge_lengths, dyn_obs, full_node_mask
        )  # (B, N+1, N+1, 8)

        return {
            "node_features":  node_features,
            "node_positions": node_positions,
            "edge_features":  edge_features,
            "node_mask":      full_node_mask,
            "edge_mask":      edges,
        }

    # ------------------------------------------------------------------
    # Step 1: 2D distance field
    # ------------------------------------------------------------------

    def _compute_distance_field(
        self,
        ray_hits_w: torch.Tensor,  # (B, N_rays, 3)
        ray_pos_w:  torch.Tensor,  # (B, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, N_rays, _ = ray_hits_w.shape
        device = ray_hits_w.device
        H = W = self.grid_size
        grid_range = self.lidar_range

        # Filter invalid rays (beyond lidar range)
        ray_dist = (ray_hits_w - ray_pos_w.unsqueeze(1)).norm(dim=-1)  # (B, N_rays)
        valid = ray_dist < (grid_range - 0.1)                          # (B, N_rays)

        # Project to horizontal plane (ego-frame)
        obs_xy = ray_hits_w[..., :2] - ray_pos_w[:, :2].unsqueeze(1)  # (B, N_rays, 2)

        # Build grid
        coords = torch.linspace(-grid_range, grid_range, H, device=device)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')  # (H, W)
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)                 # (H, W, 2)
        grid_flat = grid_xy.view(-1, 2)                                  # (H*W, 2)

        # Chunked pairwise distance: avoids materialising a full (B, H*W, N_rays) tensor.
        # Peak memory per chunk = B * chunk_size * N_rays * 4 bytes.
        # With chunk_size=500, B=128, N_rays=144 → ~37 MB (safe).
        chunk_size = 500
        dist_field_flat = torch.full((B, H * W), float('inf'), device=device)
        valid_mask = valid.unsqueeze(1)  # (B, 1, N_rays) for broadcast

        for start in range(0, H * W, chunk_size):
            end = min(start + chunk_size, H * W)
            chunk_grid = grid_flat[start:end]                              # (C, 2)
            # (B, C, 2) vs (B, N_rays, 2) → (B, C, N_rays)
            chunk_dist = torch.cdist(
                chunk_grid.unsqueeze(0).expand(B, -1, -1),
                obs_xy,
            )
            chunk_dist = chunk_dist.masked_fill(~valid_mask.expand(-1, end - start, -1),
                                                float('inf'))
            dist_field_flat[:, start:end] = chunk_dist.min(dim=-1).values
        dist_field = dist_field_flat.view(B, H, W)

        grid_res = 2.0 * grid_range / H

        # Handle inf / nan:
        # Cells with no valid ray-hit (open space beyond lidar range) must be 0.0,
        # NOT grid_range.  Filling with grid_range would create spurious "high-clearance"
        # zones behind / outside the sensor range and cause Voronoi nodes to be placed
        # there, driving the drone backwards.
        dist_field = dist_field.nan_to_num(nan=0.0, posinf=0.0)
        dist_field = dist_field.clamp(min=1e-4, max=grid_range)

        # Gradient (central difference)
        gradient = self._gradient_2d(dist_field, grid_res)  # (B, H, W, 2)

        return dist_field, gradient, grid_xy

    def _gradient_2d(self, field: torch.Tensor, resolution: float) -> torch.Tensor:
        g_x = torch.zeros_like(field)
        g_y = torch.zeros_like(field)
        g_x[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2 * resolution)
        g_y[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2 * resolution)
        g_x[:, 0, :]  = (field[:, 1, :]  - field[:, 0, :])  / resolution
        g_x[:, -1, :] = (field[:, -1, :] - field[:, -2, :]) / resolution
        g_y[:, :, 0]  = (field[:, :, 1]  - field[:, :, 0])  / resolution
        g_y[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / resolution
        return torch.stack([g_x, g_y], dim=-1)

    # ------------------------------------------------------------------
    # Step 2: Voronoi node extraction  (fully vectorized, no Python for-loops)
    # ------------------------------------------------------------------

    def _find_voronoi_nodes(
        self,
        dist_field:  torch.Tensor,           # (B, H, W)
        gradient:    torch.Tensor,           # (B, H, W, 2)
        grid_xy:     torch.Tensor,           # (H, W, 2)  ego-local frame
        ego_pos:     torch.Tensor | None = None,  # (B, 3) world frame — for goal bias
        target_pos:  torch.Tensor | None = None,  # (B, 3) world frame — for goal bias
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract Voronoi-diagram nodes from the 2-D distance field.

        Strategy:
            1. Find pixels that are local distance maxima (low gradient, high clearance).
            2. Flatten the candidate map to a fixed-size pool via a top-K by clearance
               (avoiding any Python per-batch loop).
            3. Run a vectorized greedy NMS over the top-K pool.
            4. Pack results into (B, max_nodes, …) tensors.

        Args:
            dist_field: (B, H, W) 2-D distance-to-obstacle field.
            gradient:   (B, H, W, 2) spatial gradient of dist_field.
            grid_xy:    (H, W, 2) pixel centres in ego-local XY frame.
            ego_pos:    (B, 3) world-frame ego position (optional; enables goal-direction bias).
            target_pos: (B, 3) world-frame goal position (optional; enables goal-direction bias).

        Returns:
            nodes_local : (B, max_nodes, 3)  XY in ego-local frame, Z=0.
            clearances  : (B, max_nodes)     clearance value per node.
            node_mask   : (B, max_nodes) bool  True = valid node.
        """
        B, H, W   = dist_field.shape
        device    = dist_field.device
        N         = self.max_nodes

        # -- Candidate pixels --
        dist_padded  = F.max_pool2d(dist_field.unsqueeze(1), 3, 1, 1).squeeze(1)
        is_local_max = (dist_field == dist_padded)
        grad_norm    = gradient.norm(dim=-1)
        candidates   = is_local_max & (grad_norm < self.grad_threshold) & (dist_field > self.safe_radius)
        # candidates: (B, H, W) bool

        # Flat scores: use dist_field as score; non-candidates get -inf
        scores_flat = dist_field.view(B, H * W).clone()
        scores_flat[~candidates.view(B, H * W)] = -1e9

        # Top-K pool (K = min(pool_size, H*W)); keeps best candidates per batch
        pool_size = min(N * 4, H * W)  # head room for NMS to prune from
        topk_vals, topk_idx = torch.topk(scores_flat, pool_size, dim=1)  # (B, K)

        # Position of each pool element in ego-local XY
        # grid_xy is (H, W, 2); flatten to (H*W, 2) then index
        grid_flat  = grid_xy.view(H * W, 2)                             # (H*W, 2)
        pool_xy    = grid_flat[topk_idx.view(-1)].view(B, pool_size, 2) # (B, K, 2)
        pool_xyz   = torch.cat([pool_xy,
                                 torch.zeros(B, pool_size, 1, device=device)], dim=-1)  # (B,K,3)
        pool_score = topk_vals                                           # (B, K)
        pool_valid = pool_score > -1e8                                   # (B, K) bool

        # -- Goal-direction bias: reward nodes that lie along the path to the goal.
        # Uses ego-local XY (pool_xy is already relative to ego).
        # Weight 1.5 is scaled relative to lidar_range (10 m) so that a node
        # directly ahead scores +1.5 m and directly behind scores −1.5 m.
        # This is large enough to meaningfully break ties between similarly-clear
        # Voronoi corridors while still being overpowered by a 2 m clearance gap
        # (i.e., safety dominates, but goal direction provides a real tiebreak).
        if ego_pos is not None and target_pos is not None:
            goal_xy   = target_pos[:, :2] - ego_pos[:, :2]        # (B, 2) local goal dir
            goal_n    = F.normalize(goal_xy, dim=-1)               # (B, 2) unit
            # projection of each candidate in ego-local XY onto goal direction
            proj      = (pool_xy * goal_n.unsqueeze(1)).sum(-1)    # (B, K) ∈ [-R, +R] m
            proj_n    = proj / float(self.lidar_range)             # (B, K) ∈ [-1, +1]
            goal_bonus = 1.5 * proj_n * pool_valid.float()         # ±1.5 m; only valid candidates
            pool_score = pool_score + goal_bonus

        # -- Vectorized greedy NMS over the pool --
        nodes_local, clearances, node_mask = self._nms_batched(
            pool_xyz, pool_score, pool_valid, N, self.nms_radius
        )
        return nodes_local, clearances, node_mask

    def _nms_batched(
        self,
        positions: torch.Tensor,   # (B, K, 3)
        scores:    torch.Tensor,   # (B, K)   higher = better
        valid:     torch.Tensor,   # (B, K) bool
        max_keep:  int,
        radius:    float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized greedy NMS across a whole batch.

        Iterates at most ``max_keep`` times in Python, but each iteration does a
        fully batched torch operation over all B environments.  This replaces the
        old O(B * C^2) double Python loop with O(max_keep * B) tensor ops.

        Args:
            positions: (B, K, 3) candidate positions.
            scores:    (B, K)    candidate scores (higher is better).
            valid:     (B, K) bool  which candidates are real (not padding).
            max_keep:  maximum nodes to keep per environment.
            radius:    NMS suppression radius.

        Returns:
            out_pos  : (B, max_keep, 3)
            out_score: (B, max_keep)
            out_mask : (B, max_keep) bool
        """
        B, K, _  = positions.shape
        device   = positions.device
        N        = max_keep

        out_pos   = torch.zeros(B, N, 3, device=device)
        out_score = torch.zeros(B, N,    device=device)
        out_mask  = torch.zeros(B, N,    dtype=torch.bool, device=device)

        suppressed = ~valid.clone()  # (B, K): True = should not be selected

        for slot in range(N):
            # Pick highest-score non-suppressed candidate per batch.
            # Use masked_fill to avoid a clone + index-set; also avoids the
            # GPU→CPU sync that "if not any_valid.any(): break" would cause
            # (24 syncs per env-step with the old early-exit pattern).
            best_score, best_idx = scores.masked_fill(suppressed, -1e9).max(dim=1)

            any_valid = best_score > -1e8                     # (B,) bool

            # Gather best position for each batch element
            best_pos = positions[torch.arange(B, device=device), best_idx]  # (B, 3)

            # Write into output
            out_pos[:, slot]   = torch.where(any_valid.unsqueeze(-1), best_pos,
                                              out_pos[:, slot])
            out_score[:, slot] = torch.where(any_valid, best_score,
                                              out_score[:, slot])
            out_mask[:, slot]  = any_valid

            # Suppress all candidates within radius of the chosen node (vectorized)
            dists = (positions - best_pos.unsqueeze(1)).norm(dim=-1)  # (B, K)
            suppressed = suppressed | (dists <= radius)

        return out_pos, out_score, out_mask

    # ------------------------------------------------------------------
    # Step 3: Build edges
    # ------------------------------------------------------------------

    def _build_edges(
        self,
        node_positions: torch.Tensor,  # (B, N+1, 3)
        node_mask:      torch.Tensor,  # (B, N+1) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, Np1, _ = node_positions.shape
        device = node_positions.device

        # Pairwise distances
        pi = node_positions.unsqueeze(2)   # (B, N+1, 1, 3)
        pj = node_positions.unsqueeze(1)   # (B, 1, N+1, 3)
        pdist = (pi - pj).norm(dim=-1)     # (B, N+1, N+1)

        # Valid pair mask
        mi = node_mask.unsqueeze(2)  # (B, N+1, 1)
        mj = node_mask.unsqueeze(1)  # (B, 1, N+1)
        valid_pairs = mi & mj        # (B, N+1, N+1)

        # KNN (masked)
        pdist_masked = pdist.clone()
        pdist_masked[~valid_pairs] = float('inf')
        k = min(self.k_neighbors + 1, Np1)  # +1 because self is included
        _, knn_idx = torch.topk(pdist_masked, k=k, dim=2, largest=False)  # (B, N+1, k)

        # Build adjacency (vectorized scatter)
        edges = torch.zeros(B, Np1, Np1, dtype=torch.bool, device=device)
        neighbor_idx = knn_idx[:, :, 1:]                         # skip self (dist=0)
        src = torch.arange(Np1, device=device).unsqueeze(0).unsqueeze(-1).expand(B, Np1, k - 1)
        edges.scatter_(2, neighbor_idx.clamp(0, Np1 - 1), True)

        # Symmetrize
        edges = edges | edges.transpose(1, 2)

        # Distance threshold
        edges = edges & (pdist <= self.max_edge_length) & valid_pairs

        # Remove self-loops
        eye = torch.eye(Np1, dtype=torch.bool, device=device).unsqueeze(0)
        edges = edges & ~eye

        edge_lengths = pdist * edges.float()
        return edges, edge_lengths

    # ------------------------------------------------------------------
    # Step 4: Node features (B, N+1, 18)
    # ------------------------------------------------------------------

    def _encode_node_features(
        self,
        node_positions: torch.Tensor,  # (B, N+1, 3)
        clearances:     torch.Tensor,  # (B, max_nodes)  - only for non-ego nodes
        node_mask:      torch.Tensor,  # (B, N+1) bool
        ego_pos:        torch.Tensor,  # (B, 3)
        ego_vel:        torch.Tensor,  # (B, 3)
        target_pos:     torch.Tensor,  # (B, 3)
    ) -> torch.Tensor:

        B, Np1, _ = node_positions.shape
        device = node_positions.device

        # Relative position to ego (3)
        rel_ego = node_positions - ego_pos.unsqueeze(1)  # (B, N+1, 3)

        # Relative position to target (3)
        rel_tgt = node_positions - target_pos.unsqueeze(1)  # (B, N+1, 3)

        # Clearance (1): ego node clearance = 0 (placeholder)
        ego_clr = torch.zeros(B, 1, device=device)
        clr_all = torch.cat([ego_clr, clearances], dim=1).unsqueeze(-1)  # (B, N+1, 1)

        # Distance to ego (1)
        dist_ego = rel_ego.norm(dim=-1, keepdim=True)   # (B, N+1, 1)

        # Distance to target (1)
        dist_tgt = rel_tgt.norm(dim=-1, keepdim=True)   # (B, N+1, 1)

        # Unit direction to target (3)
        dir_tgt = rel_tgt / dist_tgt.clamp(min=1e-6)    # (B, N+1, 3)

        # Ego velocity broadcast (3)
        vel_feat = ego_vel.unsqueeze(1).expand(-1, Np1, -1)  # (B, N+1, 3)

        # Ego-node flag (1): 1 for index-0, 0 otherwise
        ego_flag = torch.zeros(B, Np1, 1, device=device)
        ego_flag[:, 0, :] = 1.0

        # Valid flag (1)
        valid_feat = node_mask.float().unsqueeze(-1)  # (B, N+1, 1)

        # Normalise spatial features to O(1) scale so the initial linear projection
        # doesn't need to learn large weight downcaling.
        # lidar_range = 10m covers all within-graph distances; target can be up to 48m
        # so we use a wider context_range = 3× lidar_range for target-relative features.
        context_range   = float(self.lidar_range)       # 10 m  — for ego-relative
        global_range    = 48.0                           # map half-extent — for target-relative
        rel_ego_n  = rel_ego  / context_range           # (B, N+1, 3) ∈ [-1, +1] within lidar
        rel_tgt_n  = rel_tgt  / global_range            # (B, N+1, 3) ∈ [-1, +1] across map
        dist_ego_n = dist_ego / context_range           # (B, N+1, 1) ∈ [0, ?]; >1 outside lidar
        dist_tgt_n = dist_tgt / global_range            # (B, N+1, 1) ∈ [0, 1] across map
        vel_feat_n = vel_feat / float(self.lidar_range) # (B, N+1, 3) ~O(1) at v_max=5 m/s

        # Normalised clearance
        clr_norm = (clr_all / self.lidar_range).clamp(0, 1)  # (B, N+1, 1)

        features = torch.cat([
            rel_ego_n,     # 3  — ego-relative position, normalised
            rel_tgt_n,     # 3  — target-relative position, normalised
            clr_all,       # 1  — raw clearance (m)
            clr_norm,      # 1  — clearance / lidar_range ∈ [0,1]
            dist_ego_n,    # 1  — distance to ego, normalised
            dist_tgt_n,    # 1  — distance to target, normalised (progress signal)
            dir_tgt,       # 3  — unit direction to target (already O(1))
            vel_feat_n,    # 3  — ego velocity, normalised
            ego_flag,      # 1  — 1 for ego node, 0 otherwise
            valid_feat,    # 1  — 1 for valid node, 0 for padding
        ], dim=-1)  # (B, N+1, 18)

        # Zero out padded nodes
        features = features * node_mask.float().unsqueeze(-1)
        # Restore ego features (always valid)
        return features

    # ------------------------------------------------------------------
    # Step 5: Edge features (B, N+1, N+1, 7)
    # ------------------------------------------------------------------

    def _encode_edge_features(
        self,
        node_positions: torch.Tensor,  # (B, N+1, 3)
        edges:          torch.Tensor,  # (B, N+1, N+1) bool
        edge_lengths:   torch.Tensor,  # (B, N+1, N+1)
        dyn_obs:        torch.Tensor,  # (B, N_dyn, 10)
        node_mask:      torch.Tensor,  # (B, N+1)
    ) -> torch.Tensor:

        B, Np1, _ = node_positions.shape
        device = node_positions.device

        # Relative position vector (3)
        pi = node_positions.unsqueeze(2)   # (B, N+1, 1, 3)
        pj = node_positions.unsqueeze(1)   # (B, 1, N+1, 3)
        rel_pos = pj - pi                  # (B, N+1, N+1, 3)

        # Edge length (1)
        elen = edge_lengths.unsqueeze(-1)  # (B, N+1, N+1, 1)

        # Unit direction (3): reuse rel_pos / length
        dir_e = rel_pos / (elen.clamp(min=1e-6))  # (B, N+1, N+1, 3)

        # Nearest dynamic obstacle distance to edge midpoint (1)
        mid_pos = (pi + pj) / 2.0  # (B, N+1, N+1, 3)
        if dyn_obs.shape[1] > 0:
            obs_pos = dyn_obs[:, :, :3]           # (B, N_dyn, 3)
            # (B, N+1, N+1, N_dyn)
            diffs = mid_pos.unsqueeze(-2) - obs_pos.unsqueeze(1).unsqueeze(1)
            dyn_dists = diffs.norm(dim=-1)         # (B, N+1, N+1, N_dyn)
            min_dyn_dist = dyn_dists.min(dim=-1).values.unsqueeze(-1)  # (B, N+1, N+1, 1)
        else:
            min_dyn_dist = torch.full((B, Np1, Np1, 1), self.lidar_range, device=device)

        # Normalize
        elen_norm = (elen / self.lidar_range).clamp(0, 1)
        dyn_norm  = (min_dyn_dist / self.lidar_range).clamp(0, 1)

        # Concatenate: rel_pos(3) + elen_norm(1) + dyn_norm(1) + dir_e(3) = 8
        # NOTE: edge_feat_dim in topo.yaml must be 8 to match; was previously 7
        # because dyn_norm was computed but never included. Corrected here.
        edge_feat = torch.cat([
            rel_pos,      # 3
            elen_norm,    # 1
            dyn_norm,     # 1  ← dynamic obstacle safety distance (was dropped before)
            dir_e,        # 3
        ], dim=-1)  # (B, N+1, N+1, 8)

        # Zero out invalid edges
        edge_feat = edge_feat * edges.float().unsqueeze(-1)

        return edge_feat
