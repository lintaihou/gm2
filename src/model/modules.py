import torch
import torch.nn as nn
import torch.nn.functional as F

from .functions import apply_rope, fetch_select, scale_softmax, softplus_0_1, sparsify_triton

scale = scale_softmax
sparsify = sparsify_triton
fetch = fetch_select


class SparseLayer(nn.Module):
    def __init__(self, cfg, referral_info):
        super().__init__()
        self.norm_1 = RMSNorm(cfg.hidden_size, eps=1e-6)
        self.refs = nn.ModuleList([SparseEdgeReferral(cfg, refresh_info) for refresh_info in referral_info])
        self.att = SparseEdgeAttention(cfg)
        self.norm_2 = RMSNorm(cfg.hidden_size, eps=1e-6)
        self.ffn = SwiGLUMLP(cfg.hidden_size, int(cfg.hidden_size * cfg.expansion), cfg.hidden_size, bias=False)

    def forward(self, features, indices, weights, input_indices, input_weights, dense_indices, dense_weights):
        skip_features = features
        features = self.norm_1(features)
        for ref in self.refs:
            indices, weights = ref(features, indices, weights, input_indices, input_weights, dense_indices, dense_weights)
        features, indices, weights = self.att(features, indices, weights)

        features = features + skip_features
        skip_features = features
        features = self.norm_2(features)
        features = self.ffn(features)
        features = features + skip_features
        return features, indices, weights


class DenseLayer(nn.Module):
    def __init__(self, cfg, refresh_info):
        super().__init__()
        self.norm_1 = RMSNorm(cfg.hidden_size, eps=1e-6)
        self.att = GroupedQueryAttention(cfg, refresh_info)
        self.norm_2 = RMSNorm(cfg.hidden_size, eps=1e-6)
        self.ffn = SwiGLUMLP(cfg.hidden_size, int(cfg.hidden_size * cfg.expansion), cfg.hidden_size, bias=False)

    def forward(self, features, cos, sin):
        skip_features = features
        features = self.norm_1(features)
        features, dense_indices, dense_weights = self.att(features, cos, sin)
        features = features + skip_features

        skip_features = features
        features = self.norm_2(features)
        features = self.ffn(features)
        features = features + skip_features
        return features, dense_indices, dense_weights


class SparseEdgeReferral(nn.Module):
    def __init__(self, cfg, refresh_info):
        super().__init__()
        self.cfg = cfg
        d, k = cfg.hidden_size, cfg.n_edges
        self.r_i, self.r_d = refresh_info
        k_ = k + self.r_i + self.r_d
        self.proj = nn.Linear(d, 3 * k + 2 * k * k_, bias=False)

    def forward(self, features, indices, weights, input_indices, input_weights, dense_indices, dense_weights):
        # Define variables
        b, n, _ = features.shape
        k, r_i, r_d = self.cfg.n_edges, self.r_i, self.r_d
        k_ = k + r_i + r_d
        s_1, s_2, s_3, _ = self.cfg.sparsities

        # Project
        ins = self.proj(features)  # B, N, ...
        raw_temps, logits = ins.split([3 * k, 2 * k * k_], dim=-1)  # B, N, ...
        pre_temps, post_temps = softplus_0_1(raw_temps).split([k, 2 * k], dim=-1)  # B, N, K/2*K
        pre_temps = pre_temps.view(b, n, k, 1)  # B, N, K, 1
        post_temps = post_temps.view(b, n, 2 * k, 1).permute(0, 2, 1, 3)  # B, 2*K, N, 1
        logits = logits.view(b, n, 2 * k, k_).permute(0, 2, 1, 3)  # B, 2*K, N, K'

        # Mix
        weights = scale(weights, pre_temps)  # B, N, K, S_1
        if r_i and r_d:
            indices = torch.cat([indices, input_indices, dense_indices], dim=2)  # B, N, K', S_1
            weights = torch.cat([weights, input_weights, dense_weights], dim=2)  # B, N, K', S_1
        elif r_i:
            indices = torch.cat([indices, input_indices], dim=2)  # B, N, K', S_1
            weights = torch.cat([weights, input_weights], dim=2)  # B, N, K', S_1
        elif r_d:
            indices = torch.cat([indices, dense_indices], dim=2)  # B, N, K', S_1
            weights = torch.cat([weights, dense_weights], dim=2)  # B, N, K', S_1
        channel_weights = logits.softmax(dim=-1)  # B, 2*K, N, K'
        mixed_weights = torch.einsum("bhnk,bnkj->bhnkj", channel_weights, weights).view(b, 2 * k, n, k_ * s_1)  # B, 2*K, N, K'*S_1
        mixed_weights = scale(mixed_weights, post_temps)  # B, 2*K, N, K'*S_1
        mixed_indices = indices.unsqueeze(1).expand(-1, 2 * k, -1, -1, -1).reshape(b, 2 * k, n, k_ * s_1)  # B, 2*K, N, K'*S_1
        mixed_e1_indices, mixed_e2_indices = mixed_indices.chunk(2, dim=1)  # B, K, N, K'*S_1
        mixed_e1_weights, mixed_e2_weights = mixed_weights.chunk(2, dim=1)  # B, K, N, K'*S_1
        mixed_e1_indices, mixed_e1_weights = sparsify(mixed_e1_indices, mixed_e1_weights, s_2)  # B, K, N, S_2
        mixed_e2_indices, mixed_e2_weights = sparsify(mixed_e2_indices, mixed_e2_weights, s_3)  # B, K, N, S_3

        # Fetch
        out_indices = fetch(input=mixed_e2_indices, index=mixed_e1_indices)  # B, K, N, S_2, S_3
        fetched_e2_weights = fetch(input=mixed_e2_weights, index=mixed_e1_indices)  # B, K, N, S_2, S_3

        # Refer
        out_weights = torch.einsum("bhni,bhnij->bhnij", mixed_e1_weights, fetched_e2_weights)  # B, K, N, S_2, S_3
        indices = out_indices.view(b, k, n, s_2 * s_3).permute(0, 2, 1, 3)  # B, N, K, S_2*S_3
        weights = out_weights.view(b, k, n, s_2 * s_3).permute(0, 2, 1, 3)  # B, N, K, S_2*S_3
        indices, weights = sparsify(indices, weights, s_1)  # B, N, K, S_1
        return indices, weights


class SparseEdgeAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, k, h, d_h, g = cfg.hidden_size, cfg.n_edges, cfg.n_heads, cfg.head_size, cfg.n_groups
        self.in_proj = nn.Linear(d, k + g + h + h * d_h + g * k + g * (2 * d_h), bias=False)
        self.q_norm = RMSNorm(d_h, eps=1e-6)
        self.k_norm = RMSNorm(d_h, eps=1e-6)
        self.out_proj = nn.Linear(h * d_h, d, bias=False)

    def forward(self, features, indices, weights):
        # Define variables
        b, n, _ = features.shape
        k, h, d_h, g = self.cfg.n_edges, self.cfg.n_heads, self.cfg.head_size, self.cfg.n_groups
        s_1, _, _, s_4 = self.cfg.sparsities
        r = h // g

        # Project in
        ins = self.in_proj(features)  # B, N, ...
        raw_temps, queries, logits, keys, values = ins.split([k + g + h, h * d_h, g * k, g * d_h, g * d_h], dim=-1)  # B, N, ...
        edge_pre_temps, edge_post_temps, node_temps = softplus_0_1(raw_temps).split([k, g, h], dim=-1)  # B, N, K/G/H
        edge_pre_temps = edge_pre_temps.view(b, n, k, 1)  # B, N, K, 1
        edge_post_temps = edge_post_temps.view(b, n, g, 1).permute(0, 2, 1, 3)  # B, G, N, 1
        node_temps = node_temps.view(b, n, h, 1).permute(0, 2, 1, 3).unflatten(1, (g, r))  # B, G, R, N, 1
        queries = queries.view(b, n, h, d_h).permute(0, 2, 1, 3).unflatten(1, (g, r))  # B, G, R, N, D_h
        logits = logits.view(b, n, g, k).permute(0, 2, 1, 3)  # B, G, N, K
        keys = keys.view(b, n, g, d_h).permute(0, 2, 1, 3)  # B, G, N, D_h
        values = values.view(b, n, g, d_h).permute(0, 2, 1, 3)  # B, G, N, D_h
        queries = self.q_norm(queries)  # B, G, R, N, D_h
        keys = self.k_norm(keys)  # B, G, N, D_h
        keys_values = torch.cat([keys, values], dim=-1)  # B, G, N, 2*D_h

        # Mix
        weights = scale(weights, edge_pre_temps)  # B, N, K, S_1
        channel_weights = logits.softmax(dim=-1)  # B, G, N, K
        mixed_weights = torch.einsum("bhnk,bnkj->bhnkj", channel_weights, weights).view(b, g, n, k * s_1)  # B, G, N, K*S_1
        mixed_indices = indices.unsqueeze(1).expand(-1, g, -1, -1, -1).reshape(b, g, n, k * s_1)  # B, G, N, K*S_1
        mixed_indices, mixed_weights = sparsify(mixed_indices, mixed_weights, s_4)  # B, G, N, S_4

        # Fetch
        fetched_keys_values = fetch(input=keys_values, index=mixed_indices)  # B, G, N, S_4, 2*D_h
        fetched_keys, fetched_values = fetched_keys_values.chunk(2, dim=-1)  # B, G, N, S_4, D_h

        # Attend
        edge_factors = mixed_weights.clamp_min(1e-8).log() * edge_post_temps  # B, G, N, S_4
        node_factors = torch.einsum("bgrnd,bgnid->bgrni", queries, fetched_keys) * node_temps / d_h**0.5  # B, G, R, N, S_4
        attention_scores = edge_factors.unsqueeze(2) + node_factors  # B, G, R, N, S_4
        attention_weights = attention_scores.softmax(dim=-1)  # B, G, R, N, S_4
        outs = torch.einsum("bgrni,bgnid->bngrd", attention_weights, fetched_values)  # B, N, G, R, D_h

        # Project out
        features = self.out_proj(outs.flatten(2))  # B, N, D

        # Realign
        if self.cfg.realign:
            if k > h:
                weights = torch.cat([weights[:, :, :-h], attention_weights.flatten(1, 2).permute(0, 2, 1, 3)], dim=-2)  # B, N, K, S_1
                indices = torch.cat([indices[:, :, :-h], mixed_indices.repeat_interleave(r, dim=1).permute(0, 2, 1, 3)], dim=-2)  # B, N, K, S_1
            else:  # k <= h:
                weights = attention_weights.transpose(1, 2).flatten(1, 2).permute(0, 2, 1, 3)[:, :, :k]  # B, N, K, S_1
                indices = mixed_indices.repeat(1, r, 1, 1).permute(0, 2, 1, 3)[:, :, :k]  # B, N, K, S_1
        return features, indices, weights


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, refresh_info):
        super().__init__()
        self.cfg = cfg
        self.r_d = refresh_info
        d, h, g, d_h = cfg.hidden_size, cfg.n_heads, cfg.n_groups, cfg.head_size
        self.qkv_proj = nn.Linear(d, h * d_h + 2 * g * d_h, bias=False)
        self.q_norm = RMSNorm(d_h, eps=1e-6)
        self.k_norm = RMSNorm(d_h, eps=1e-6)
        self.o_proj = nn.Linear(h * d_h, d, bias=False)

    def forward(self, features, cos, sin):
        # Define variables
        b, n, _ = features.shape
        h, g, d_h, r_d = self.cfg.n_heads, self.cfg.n_groups, self.cfg.head_size, self.r_d
        s_1, _, _, _ = self.cfg.sparsities

        # Project in
        queries_keys_values = self.qkv_proj(features)  # B, N, ...
        queries, keys, values = queries_keys_values.split([h * d_h, g * d_h, g * d_h], dim=-1)  # B, N, ...
        queries = queries.view(b, n, h, d_h).transpose(1, 2)  # B, H, N, D_h
        keys = keys.view(b, n, g, d_h).transpose(1, 2)  # B, G, N, D_h
        values = values.view(b, n, g, d_h).transpose(1, 2)  # B, G, N, D_h
        queries = self.q_norm(queries)  # B, H, N, D_h
        keys = self.k_norm(keys)  # B, G, N, D_h
        queries = apply_rope(queries, cos, sin)  # B, H, N, D_h
        keys = apply_rope(keys, cos, sin)  # B, G, N, D_h

        # Attend
        features = F.scaled_dot_product_attention(queries, keys, values, is_causal=True, enable_gqa=True)  # B, H, N, D_h
        if r_d:
            reordered_queries = queries.unflatten(1, (g, h // g)).transpose(1, 2).flatten(1, 2)[:, -r_d:]  # B, R_d, N, D_h
            expanded_keys = keys.repeat(1, h // g, 1, 1)[:, -r_d:]  # B, R_d, N, D_h
            scores = (reordered_queries @ expanded_keys.transpose(-2, -1)) / d_h**0.5  # B, R_d, N, N
            causal_mask = torch.ones(n, n, device=scores.device, dtype=torch.bool).triu(1)  # N, N
            scores.masked_fill_(causal_mask, float("-inf"))  # B, R_d, N, N
            dense_scores, dense_indices = scores.topk(s_1, dim=-1)  # B, R_d, N, S_1
            dense_indices = dense_indices.masked_fill(dense_scores.isneginf(), 0)  # B, R_d, N, S_1
            dense_weights = dense_scores.softmax(dim=-1)  # B, R_d, N, S_1
            dense_indices = dense_indices.transpose(1, 2)  # B, N, R_d, S_1
            dense_weights = dense_weights.transpose(1, 2)  # B, N, R_d, S_1
        else:
            dense_indices, dense_weights = None, None

        # Project out
        features = features.transpose(1, 2).contiguous().view(b, n, h * d_h)  # B, N, H*D_h
        features = self.o_proj(features)  # B, N, D
        return features, dense_indices, dense_weights


class SwiGLUMLP(nn.Module):
    def __init__(self, in_size, inter_size, out_size, bias):
        super().__init__()
        self.gate_up_proj = nn.Linear(in_size, inter_size * 2, bias=bias)
        self.down_proj = nn.Linear(inter_size, out_size, bias=bias)

    def forward(self, features):
        gates, ups = self.gate_up_proj(features).chunk(2, dim=-1)
        return self.down_proj(F.silu(gates) * ups)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, features):
        input_dtype = features.dtype
        features = features.to(torch.float32)
        variance = features.square().mean(dim=-1, keepdim=True)
        features = features * torch.rsqrt(variance + self.eps)
        return self.weight * features.to(input_dtype)
