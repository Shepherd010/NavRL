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
                 use_spd_bias: bool = True):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim  = hidden_dim
        self.num_heads   = num_heads
        self.head_dim    = hidden_dim // num_heads
        self.use_spd_bias = use_spd_bias

        # QKV projections
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Topology bias networks
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
    ) -> torch.Tensor:

        B, N, _ = h.shape

        # -- Multi-head attention --
        Q = self.W_Q(h).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, d)
        K = self.W_K(h).view(B, N, self.num_heads, self.head_dim)
        V = self.W_V(h).view(B, N, self.num_heads, self.head_dim)

        # Attention scores: (B, H, N, N)
        scores = torch.einsum('bihd,bjhd->bhij', Q, K) / np.sqrt(self.head_dim)

        # -- Topology bias --
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
        # rows from invalid nodes should not attend; cols to invalid nodes masked
        node_mask_exp = node_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, N, -1)
        scores = scores.masked_fill(~node_mask_exp, -1e9)

        attn = F.softmax(scores, dim=-1)  # (B, H, N, N)

        # -- Aggregate --
        out = torch.einsum('bhij,bjhd->bihd', attn, V)    # (B, N, H, d)
        out = out.contiguous().view(B, N, self.hidden_dim)
        out = self.drop1(self.W_O(out))

        # -- Residual + LayerNorm --
        h = self.norm1(h + out)
        h = self.norm2(h + self.drop2(self.ffn(h)))
        return h


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
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim   = hidden_dim
        self.num_heads    = num_heads
        self.num_layers   = num_layers
        self.use_spd_bias = use_spd_bias

        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout, use_spd_bias)
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
        node_features: torch.Tensor,   # (B, N+1, 18)
        edge_features: torch.Tensor,   # (B, N+1, N+1, 7)
        node_mask:     torch.Tensor,   # (B, N+1) bool
        edge_mask:     torch.Tensor,   # (B, N+1, N+1) bool
    ):
        B, Np1, _ = node_features.shape

        # Input projections
        h = self.node_proj(node_features)  # (B, N+1, D)
        e = self.edge_proj(edge_features)  # (B, N+1, N+1, D)

        # Precompute SPD matrix once (shared across all layers)
        spd_matrix = self._compute_spd_matrix(edge_mask) if self.use_spd_bias else None

        # Transformer layers
        for layer in self.layers:
            h = layer(h, e, node_mask, edge_mask, spd_matrix)

        # -- Node selection head --
        h_ego  = h[:, 0:1, :]                                    # (B, 1, D)
        h_cand = h[:, 1:, :]                                      # (B, N, D)
        h_ego_exp = h_ego.expand(-1, Np1 - 1, -1)                # (B, N, D)
        h_concat  = torch.cat([h_ego_exp, h_cand], dim=-1)        # (B, N, 2D)
        logits    = self.score_mlp(h_concat).squeeze(-1)          # (B, N)

        # Mask invalid candidate nodes
        cand_mask = node_mask[:, 1:]   # (B, N)
        no_cand = cand_mask.sum(dim=-1) == 0
        if torch.any(no_cand):
            # Fallback must only patch rows with zero valid candidates.
            # Using cand_mask[:, 0] = True would wrongly bias all envs in the batch.
            cand_mask = cand_mask.clone()
            cand_mask[no_cand, 0] = True

        logits = logits.masked_fill(~cand_mask, -1e9)
        probs  = F.softmax(logits, dim=-1)  # (B, N)

        return probs, h

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
