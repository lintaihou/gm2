import torch.nn as nn
import torch.nn.functional as F

from .functions import compute_edge_cache, compute_rope_cache, schedule_layers
from .modules import DenseLayer, RMSNorm, SparseLayer


class TheiaHyperionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        assert cfg.n_heads % cfg.n_groups == 0
        assert max(cfg.sparsities) <= cfg.seq_length
        if cfg.realign:
            assert cfg.sparsities[0] == cfg.sparsities[3]
        assert cfg.n_input_refreshes <= cfg.n_edges
        assert cfg.n_dense_refreshes <= cfg.n_heads

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([{0: SparseLayer, 1: DenseLayer}[t](cfg, info) for t, info in schedule_layers(cfg)])
        self.out_norm = RMSNorm(cfg.hidden_size, eps=1e-6)

        if cfg.n_layers[0]:
            input_indices, input_weights = compute_edge_cache(cfg)
            self.register_buffer("input_indices", input_indices, persistent=False)
            self.register_buffer("input_weights", input_weights, persistent=False)
        if cfg.n_layers[1]:
            cos, sin = compute_rope_cache(cfg)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=cfg.init_std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=cfg.init_std)

        self.apply(init_weights)

    def forward(self, input_ids, labels):
        features = self.embed(input_ids)
        if self.cfg.n_layers[0]:
            indices, weights = self.input_indices, self.input_weights
            start = self.input_indices.size(-2) - self.cfg.n_input_refreshes
            input_indices, input_weights = self.input_indices[..., start:, :], self.input_weights[..., start:, :]
            dense_indices, dense_weights = None, None

        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                features, dense_indices, dense_weights = layer(features, self.cos, self.sin)
            else:
                features, indices, weights = layer(features, indices, weights, input_indices, input_weights, dense_indices, dense_weights)

        shifted_logits = F.linear(self.out_norm(features[:, :-1, :]), self.embed.weight)
        shifted_labels = labels[:, 1:]

        loss = F.cross_entropy(shifted_logits.flatten(0, 1), shifted_labels.flatten())
        return loss
