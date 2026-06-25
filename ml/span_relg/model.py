import math

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import pair_geometry_dim


def pair_geometry_tensor(boxes):
    # boxes: [B, N, 4] in 0..1 unit coords.
    x0, y0, x1, y1 = [boxes[..., i] for i in range(4)]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0).clamp_min(0.0)
    h = (y1 - y0).clamp_min(0.0)
    cxi, cxj = cx.unsqueeze(2), cx.unsqueeze(1)
    cyi, cyj = cy.unsqueeze(2), cy.unsqueeze(1)
    wi, wj = w.unsqueeze(2), w.unsqueeze(1)
    hi, hj = h.unsqueeze(2), h.unsqueeze(1)
    xi0, xj0 = x0.unsqueeze(2), x0.unsqueeze(1)
    xi1, xj1 = x1.unsqueeze(2), x1.unsqueeze(1)
    yi0, yj0 = y0.unsqueeze(2), y0.unsqueeze(1)
    yi1, yj1 = y1.unsqueeze(2), y1.unsqueeze(1)
    dx = cxj - cxi
    dy = cyj - cyi
    abs_dx = dx.abs()
    abs_dy = dy.abs()
    dist = torch.sqrt(dx * dx + dy * dy + 1e-12)
    angle = torch.atan2(dy, dx)
    x_overlap = (torch.minimum(xi1, xj1) - torch.maximum(xi0, xj0)).clamp_min(0.0)
    y_overlap = (torch.minimum(yi1, yj1) - torch.maximum(yi0, yj0)).clamp_min(0.0)
    x_overlap_ratio = x_overlap / torch.minimum(wi, wj).clamp_min(1e-6)
    y_overlap_ratio = y_overlap / torch.minimum(hi, hj).clamp_min(1e-6)
    x_gap = (torch.maximum(xi0, xj0) - torch.minimum(xi1, xj1)).clamp_min(0.0)
    y_gap = (torch.maximum(yi0, yj0) - torch.minimum(yi1, yj1)).clamp_min(0.0)
    area_i = wi * hi
    area_j = wj * hj
    same_row = 1.0 / (1.0 + abs_dy / ((hi + hj) / 2.0).clamp_min(1e-6))
    same_col = 1.0 / (1.0 + abs_dx / ((wi + wj) / 2.0).clamp_min(1e-6))
    return torch.stack(
        [
            cxi.expand_as(dx),
            cyi.expand_as(dx),
            wi.expand_as(dx),
            hi.expand_as(dx),
            cxj.expand_as(dx),
            cyj.expand_as(dx),
            wj.expand_as(dx),
            hj.expand_as(dx),
            dx,
            dy,
            abs_dx,
            abs_dy,
            dist,
            torch.sin(angle),
            torch.cos(angle),
            x_overlap_ratio,
            y_overlap_ratio,
            x_gap,
            y_gap,
            (cxj > cxi).float(),
            (cxj < cxi).float(),
            (cyj > cyi).float(),
            (cyj < cyi).float(),
            area_i.expand_as(dx),
            area_j.expand_as(dx),
            area_j / area_i.clamp_min(1e-6),
            same_row,
            same_col,
        ],
        dim=-1,
    )


class SpatialPairFeatureEncoder(nn.Module):
    def __init__(self, geom_dim, num_heads, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, pair_geom):
        return self.mlp(pair_geom).permute(0, 3, 1, 2)


class SpanSpatialSelfAttentionLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.spatial = SpatialPairFeatureEncoder(pair_geometry_dim(), num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, boxes, mask):
        batch, n_nodes, _ = x.shape
        q = self.q(x).view(batch, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = logits + self.spatial(pair_geometry_tensor(boxes))
        key_mask = mask[:, None, None, :].bool()
        logits = logits.masked_fill(~key_mask, -1e4)
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch, n_nodes, self.d_model)
        x = self.norm1(x + self.dropout(self.out(context)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x * mask.unsqueeze(-1).float()


class SpanContextEncoder(nn.Module):
    def __init__(self, d_model=256, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [SpanSpatialSelfAttentionLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, boxes, mask):
        for layer in self.layers:
            x = layer(x, boxes, mask)
        return x


class RelGScorer(nn.Module):
    def __init__(self, d_model=256, geom_dim=None, dropout=0.1):
        super().__init__()
        geom_dim = geom_dim or pair_geometry_dim()
        self.head_proj = nn.Linear(d_model, d_model)
        self.dep_proj = nn.Linear(d_model, d_model)
        self.biaffine = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.biaffine)
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_model * 4 + geom_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, z, boxes, candidate_pairs):
        if candidate_pairs.numel() == 0:
            return z.new_empty((0,))
        batch_idx = candidate_pairs[:, 0].long()
        head_idx = candidate_pairs[:, 1].long()
        dep_idx = candidate_pairs[:, 2].long()
        head = self.head_proj(z[batch_idx, head_idx])
        dep = self.dep_proj(z[batch_idx, dep_idx])
        biaffine = (head @ self.biaffine * dep).sum(dim=-1)
        geom_all = pair_geometry_tensor(boxes)
        geom = geom_all[batch_idx, head_idx, dep_idx]
        pair_input = torch.cat([head, dep, head * dep, (head - dep).abs(), geom], dim=-1)
        return biaffine + self.pair_mlp(pair_input).squeeze(-1)


class SpanRelGModel(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_fields,
        num_kinds,
        d_model=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_dim, d_model)
        self.field_emb = nn.Embedding(num_fields, d_model)
        self.kind_emb = nn.Embedding(num_kinds, d_model)
        self.box_mlp = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = SpanContextEncoder(d_model, num_layers, num_heads, dropout)
        self.scorer = RelGScorer(d_model, pair_geometry_dim(), dropout)

    def forward(
        self,
        node_hidden,
        node_field_ids,
        node_kind_ids,
        node_boxes,
        node_mask,
        candidate_pairs,
        pair_labels=None,
        pos_weight=None,
        pair_loss_weights=None,
    ):
        x = (
            self.hidden_proj(node_hidden)
            + self.field_emb(node_field_ids)
            + self.kind_emb(node_kind_ids)
            + self.box_mlp(node_boxes)
        )
        x = self.dropout(self.input_norm(x))
        z = self.encoder(x, node_boxes, node_mask)
        logits = self.scorer(z, node_boxes, candidate_pairs)
        output = {"logits": logits, "probs": torch.sigmoid(logits)}
        if pair_labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                pair_labels.float(),
                pos_weight=pos_weight,
                reduction="none" if pair_loss_weights is not None else "mean",
            )
            if pair_loss_weights is not None:
                weights = pair_loss_weights.float().to(loss.device)
                loss = (loss * weights).sum() / weights.sum().clamp_min(1e-6)
            output["loss"] = loss
        return output
