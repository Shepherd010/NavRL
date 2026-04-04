"""
GraphTransformer: topology-aware attention policy network.

Pure PyTorch, no Isaac Sim dependency.

Input (from TopoExtractor):
    node_features  : (B, N+1, 18)
    edge_features  : (B, N+1, N+1, 7)
    node_mask      : (B, N+1) bool
    edge_mask      : (B, N+1, N+1) bool

Output:
    probs : (B, N)     node selection probabilities (ego excluded)
    h     : (B, N+1, hidden_dim)  all node embeddings
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single Transformer layer
# ---------------------------------------------------------------------------

class GraphTransformerLayer(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1,
                 use_spd_bias: bool = True, use_topo_bias: bool = True,
                 sparse_topk: int = 0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim  = hidden_dim
        self.num_heads   = num_heads
        self.head_dim    = hidden_dim // num_heads
        self.use_spd_bias = use_spd_bias
        self.use_topo_bias = use_topo_bias
        self.sparse_topk = int(sparse_topk)

        # QKV projections
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Topology bias networks (optional; disabled by default for lower complexity)
        if use_topo_bias:
            self.edge_bias_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 4, num_heads),
            )
            if use_spd_bias:
                self.spd_bias_mlp = nn.Sequential(
                    nn.Linear(1, hidden_dim // 4),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim // 4, num_heads),
                )

        # Output projection
        self.W_O    = nn.Linear(hidden_dim, hidden_dim)
        self.drop1  = nn.Dropout(dropout)
        self.norm1  = nn.LayerNorm(hidden_dim)
        self.norm2  = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight, gain=1.0 / np.sqrt(2))
        nn.init.xavier_uniform_(self.W_K.weight, gain=1.0 / np.sqrt(2))
        nn.init.xavier_uniform_(self.W_V.weight, gain=1.0 / np.sqrt(2))
        nn.init.xavier_uniform_(self.W_O.weight)

    def forward(
        self,
        h:          torch.Tensor,                  # (B, N, hidden_dim)
        e:          torch.Tensor,                  # (B, N, N, hidden_dim)
        node_mask:  torch.Tensor,                  # (B, N) bool
        edge_mask:  torch.Tensor,                  # (B, N, N) bool
        spd_matrix: torch.Tensor | None = None,    # (B, N, N)
        force_dense: bool = False,
    ) -> torch.Tensor:
        if self.sparse_topk > 0 and not force_dense:
            return self._forward_sparse(h, e, node_mask, edge_mask, spd_matrix)
        return self._forward_dense(h, e, node_mask, edge_mask, spd_matrix)

    def _forward_dense(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        spd_matrix: torch.Tensor | None,
    ):
        B, N, _ = h.shape

        # -- Multi-head attention --
        Q = self.W_Q(h).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, d)
        K = self.W_K(h).view(B, N, self.num_heads, self.head_dim)
        V = self.W_V(h).view(B, N, self.num_heads, self.head_dim)

        # Attention scores: (B, H, N, N)
        scores = torch.einsum('bihd,bjhd->bhij', Q, K) / np.sqrt(self.head_dim)

        # -- Topology bias (optional) --
        if self.use_topo_bias:
            edge_bias = self.edge_bias_mlp(e)           # (B, N, N, H)
            edge_bias = edge_bias.permute(0, 3, 1, 2)  # (B, H, N, N)

            if self.use_spd_bias and spd_matrix is not None:
                spd_n = (spd_matrix.clamp(0, 10) / 10.0).unsqueeze(-1)  # (B, N, N, 1)
                spd_bias = self.spd_bias_mlp(spd_n).permute(0, 3, 1, 2) # (B, H, N, N)
                edge_mask_exp = edge_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                bias = torch.where(edge_mask_exp, edge_bias, spd_bias)
            else:
                bias = edge_bias
            scores = scores + bias

        # -- Node mask: pad invalid nodes --
        node_mask_exp = node_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, N, -1)
        scores = scores.masked_fill(~node_mask_exp, -1e9)

        attn = F.softmax(scores, dim=-1)  # (B, H, N, N)

        # -- Aggregate --
        out = torch.einsum('bhij,bjhd->bihd', attn, V)    # (B, N, H, d)
        out = out.contiguous().view(B, N, self.hidden_dim)
        out = self.drop1(self.W_O(out))

        h = self.norm1(h + out)
        h = self.norm2(h + self.drop2(self.ffn(h)))
        return h, attn

    def _forward_sparse(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        spd_matrix: torch.Tensor | None,
    ):
        # Sparse attention complexity: O(B * H * N * K * d), K << N.
        B, N, _ = h.shape
        k = min(self.sparse_topk, N)

        # Q/K/V in head-first layout for fast batched operations
        Qn = self.W_Q(h).view(B, N, self.num_heads, self.head_dim)
        Kn = self.W_K(h).view(B, N, self.num_heads, self.head_dim)
        Vn = self.W_V(h).view(B, N, self.num_heads, self.head_dim)
        Q = Qn.permute(0, 2, 1, 3)  # (B, H, N, d)

        device = h.device
        idx_n = torch.arange(N, device=device)

        # Allow keys: graph edges + self + ego-node(0), and only valid nodes.
        allowed = edge_mask.clone()
        allowed[:, idx_n, idx_n] = True
        allowed[:, :, 0] = True
        allowed = allowed & node_mask.unsqueeze(1)

        # Metric for top-k key selection: shortest-path first if available, else edge-first.
        if spd_matrix is not None:
            metric = spd_matrix.float().clone()
            metric = metric.masked_fill(~allowed, 1e6)
        else:
            metric = torch.where(
                allowed,
                torch.zeros((B, N, N), device=device, dtype=torch.float32),
                torch.full((B, N, N), 1e6, device=device, dtype=torch.float32),
            )
        # Deterministic tie-break by key index
        metric = metric + idx_n.view(1, 1, N).float() * 1e-4
        nn_idx = metric.topk(k, dim=-1, largest=False).indices  # (B, N, K)

        # Gather selected K/V: (B, N, K, H, d) -> (B, H, N, K, d)
        b_idx = torch.arange(B, device=device)[:, None, None]
        q_idx = idx_n.view(1, N, 1).expand(B, -1, k)
        K_sel = Kn[b_idx, nn_idx].permute(0, 3, 1, 2, 4)
        V_sel = Vn[b_idx, nn_idx].permute(0, 3, 1, 2, 4)

        # Sparse attention scores: (B, H, N, K)
        scores = (Q.unsqueeze(3) * K_sel).sum(dim=-1) / np.sqrt(self.head_dim)

        if self.use_topo_bias:
            # Selected edge bias only (avoid full NxN MLP in sparse mode)
            e_sel = e[b_idx, q_idx, nn_idx]                 # (B, N, K, D)
            edge_bias = self.edge_bias_mlp(e_sel).permute(0, 3, 1, 2)  # (B, H, N, K)
            if self.use_spd_bias and spd_matrix is not None:
                spd_sel = spd_matrix[b_idx, q_idx, nn_idx].unsqueeze(-1)          # (B, N, K, 1)
                spd_n = spd_sel.clamp(0, 10) / 10.0
                spd_bias = self.spd_bias_mlp(spd_n).permute(0, 3, 1, 2)            # (B, H, N, K)
                edge_sel = edge_mask[b_idx, q_idx, nn_idx].unsqueeze(1)            # (B, 1, N, K)
                bias = torch.where(edge_sel, edge_bias, spd_bias)
            else:
                bias = edge_bias
            scores = scores + bias

        # Mask invalid selected keys (should rarely trigger because selection uses allowed).
        key_valid_sel = node_mask[b_idx, nn_idx].unsqueeze(1)  # (B, 1, N, K)
        scores = scores.masked_fill(~key_valid_sel, -1e9)

        attn_k = F.softmax(scores, dim=-1)                     # (B, H, N, K)
        out = (attn_k.unsqueeze(-1) * V_sel).sum(dim=3)        # (B, H, N, d)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.hidden_dim)
        out = self.drop1(self.W_O(out))

        h = self.norm1(h + out)
        h = self.norm2(h + self.drop2(self.ffn(h)))

        # Training path doesn't need dense attn matrix; return None to avoid O(N^2) materialization.
        return h, None


# ---------------------------------------------------------------------------
# Full GraphTransformer
# ---------------------------------------------------------------------------

class GraphTransformer(nn.Module):
    """
    Graph Transformer policy network.

    Forward signature:
        probs, h = model(node_features, edge_features, node_mask, edge_mask)

    probs[:, k] is the probability of selecting the (k+1)-th node (0-indexed in
    node_positions), i.e. the ego node (index 0) is excluded from the action space.
    """

    def __init__(
        self,
        node_feat_dim: int = 18,
        edge_feat_dim: int = 7,
        hidden_dim:    int = 64,
        num_heads:     int = 4,
        num_layers:    int = 3,
        dropout:       float = 0.1,
        use_spd_bias:  bool = True,
        use_topo_bias: bool = True,
        sparse_topk:   int = 0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim   = hidden_dim
        self.num_heads    = num_heads
        self.num_layers   = num_layers
        self.use_spd_bias = use_spd_bias
        self.use_topo_bias = use_topo_bias
        self.sparse_topk = int(sparse_topk)

        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim, num_heads, dropout,
                use_spd_bias=use_spd_bias,
                use_topo_bias=use_topo_bias,
                sparse_topk=sparse_topk,
            )
            for _ in range(num_layers)
        ])

        # Scoring MLP: concatenate [h_ego || h_cand] -> scalar
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.zeros_(self.node_proj.bias)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.zeros_(self.edge_proj.bias)
        for m in self.score_mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        node_features: torch.Tensor,        # (B, N+1, 18)
        edge_features: torch.Tensor,        # (B, N+1, N+1, 7)
        node_mask:     torch.Tensor,        # (B, N+1) bool
        edge_mask:     torch.Tensor,        # (B, N+1, N+1) bool
        spd_matrix:    torch.Tensor = None, # (B, N+1, N+1) precomputed — pass to skip recomputation
    ):
        B, Np1, _ = node_features.shape

        # Input projections
        h = self.node_proj(node_features)  # (B, N+1, D)
        e = self.edge_proj(edge_features) if self.use_topo_bias else None  # (B, N+1, N+1, D)

        # Precompute SPD matrix once (shared across all layers)
        # Caller may pass a precomputed matrix (e.g. cached from rollout) to skip Floyd-Warshall.
        if spd_matrix is None and self.use_spd_bias:
            spd_matrix = self._compute_spd_matrix(edge_mask) if self.use_spd_bias else None

        # Transformer layers — each returns (h, attn); we discard attn in normal training
        for layer in self.layers:
            h, _ = layer(h, e, node_mask, edge_mask, spd_matrix)

        probs, cand_mask = self._score_nodes(h, node_mask, Np1)
        return probs, h

    def forward_with_attn(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask:     torch.Tensor,
        edge_mask:     torch.Tensor,
        spd_matrix:    torch.Tensor = None,
    ):
        """Same as forward() but also returns attn weights from all layers.
        Used exclusively for visualization — not called during training.
        Returns: probs (B,N), h (B,N+1,D), all_attn list[L × (B,H,N,N)]
        """
        B, Np1, _ = node_features.shape
        h = self.node_proj(node_features)
        # Keep dense path for visualization fidelity (returns full NxN attention).
        e = self.edge_proj(edge_features) if self.use_topo_bias else None
        if spd_matrix is None and self.use_spd_bias:
            spd_matrix = self._compute_spd_matrix(edge_mask) if self.use_spd_bias else None

        all_attn = []
        for layer in self.layers:
            h, attn = layer(h, e, node_mask, edge_mask, spd_matrix, force_dense=True)
            all_attn.append(attn.cpu())  # keep on CPU immediately to save VRAM

        probs, _ = self._score_nodes(h, node_mask, Np1)
        return probs, h, all_attn

    def _score_nodes(self, h, node_mask, Np1):
        """Shared scoring head used by both forward() and forward_with_attn()."""
        h_ego  = h[:, 0:1, :]
        h_cand = h[:, 1:, :]
        h_ego_exp = h_ego.expand(-1, Np1 - 1, -1)
        h_concat  = torch.cat([h_ego_exp, h_cand], dim=-1)
        logits    = self.score_mlp(h_concat).squeeze(-1)

        cand_mask = node_mask[:, 1:]
        no_cand   = cand_mask.sum(dim=-1) == 0
        if torch.any(no_cand):
            cand_mask = cand_mask.clone()
            cand_mask[no_cand, 0] = True

        logits = logits.masked_fill(~cand_mask, -1e9)
        probs  = F.softmax(logits, dim=-1)
        return probs, cand_mask

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_spd_matrix(edge_mask: torch.Tensor) -> torch.Tensor:
        """Vectorized Floyd-Warshall. O(N^3) matrix ops, no Python inner loop."""
        B, N, _ = edge_mask.shape
        device  = edge_mask.device

        spd = torch.full((B, N, N), float(N), device=device)
        spd[edge_mask] = 1.0
        idx = torch.arange(N, device=device)
        spd[:, idx, idx] = 0.0

        for k in range(N):
            spd_ik = spd[:, :, k:k+1]    # (B, N, 1)
            spd_kj = spd[:, k:k+1, :]    # (B, 1, N)
            spd = torch.minimum(spd, spd_ik + spd_kj)

        return spd

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def need_spd_matrix(self) -> bool:
        # SPD is only used when BOTH topo_bias and spd_bias are enabled.
        # With use_topo_bias=false, the SPD MLP doesn't exist — skip Floyd-Warshall.
        return bool(self.use_spd_bias and self.use_topo_bias)
