"""Causal GPT with shared token-value embeddings injected into selected blocks."""
import torch
from torch import nn
from torch.nn import functional as F

from model import Block, GPT, make_norm


class ValueBlock(Block):
    def __init__(self, width, heads, norm, mlp, use_value, initial_scale):
        super().__init__(width, heads, norm, mlp)
        self.use_value = use_value
        if use_value:
            self.value_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, x, token_value=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        if self.use_value:
            if token_value is None:
                raise ValueError('Selected value layer requires token_value.')
            extra = token_value.view(
                batch, length, self.heads, width // self.heads
            ).transpose(1, 2)
            v = v + self.value_scale * extra
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class ValueBigramGPT(GPT):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        norm = config.get('norm', 'layernorm')
        mlp = config.get('mlp', 'gelu')
        selected = set(config.get('value_layers', []))
        if not selected or min(selected) < 0 or max(selected) >= config['depth']:
            raise ValueError('value_layers must select valid transformer blocks.')
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([
            ValueBlock(
                width, config['heads'], norm, mlp, index in selected,
                config.get('value_initial_scale', 1.0),
            )
            for index in range(config['depth'])
        ])
        self.norm = make_norm(width, norm)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.value_embedding = nn.Embedding(config['vocab'], width)
        self.apply(self.initialize)
        if config.get('tie_embeddings', True):
            self.head.weight = self.token.weight

        self.bigram_enabled = config.get('bigram_enabled', True)
        self.bigram_scale = float(config.get('bigram_scale', 1.0))
        self.bigram_prior_strength = float(config.get('bigram_prior_strength', 80.0))
        self.bigram_clip = float(config.get('bigram_clip', 4.0))
        if self.bigram_enabled:
            self.bigram = nn.Embedding(config['vocab'], config['vocab'])
            nn.init.zeros_(self.bigram.weight)
            self.bigram.weight.requires_grad_(False)
            self.register_buffer(
                '_bigram_counts',
                self.bigram.weight.new_zeros(config['vocab'], config['vocab']),
                persistent=False,
            )

    @torch.no_grad()
    def observe_batch(self, inputs, targets):
        if not self.bigram_enabled:
            return
        valid = targets >= 0
        pairs = inputs[valid] * self.config['vocab'] + targets[valid]
        flat_counts = self._bigram_counts.view(-1)
        flat_counts.index_add_(0, pairs, torch.ones_like(pairs, dtype=flat_counts.dtype))

    @torch.no_grad()
    def finalize_training(self):
        if not self.bigram_enabled:
            return
        counts = self._bigram_counts
        vocab = self.config['vocab']
        row_counts = counts.sum(dim=1, keepdim=True)
        target_counts = counts.sum(dim=0)
        unigram = (target_counts + 1.0) / (target_counts.sum() + vocab)
        conditional = (
            counts + self.bigram_prior_strength * unigram
        ) / (row_counts + self.bigram_prior_strength)
        residual = (conditional.log() - unigram.log()).clamp(
            -self.bigram_clip, self.bigram_clip
        )
        self.bigram.weight.copy_(residual)

    def features(self, ids):
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        token_value = self.value_embedding(ids)
        for block in self.blocks:
            x = block(x, token_value if block.use_value else None)
        return self.norm(x)

    def forward(self, ids):
        logits = self.head(self.features(ids))
        if self.bigram_enabled:
            logits = logits + self.bigram_scale * self.bigram(ids)
        return logits


def build_model(config):
    return ValueBigramGPT(config)
