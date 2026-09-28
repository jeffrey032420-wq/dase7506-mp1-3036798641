"""A9 predictor augmented with a strictly causal shared token-value path."""
import torch
from torch import nn

from model import GPT
from student_unigram import UnigramAdaptiveGPT
from student_value import ValueBlock


class ValueUnigramAdaptiveGPT(UnigramAdaptiveGPT):
    """Preserve A9 calibration while injecting token values into chosen blocks."""

    def __init__(self, config):
        super().__init__(config)
        selected = set(config.get('value_layers', []))
        if not selected or min(selected) < 0 or max(selected) >= config['depth']:
            raise ValueError('value_layers must select valid transformer blocks.')
        width = config['width']
        norm = config.get('norm', 'layernorm')
        mlp = config.get('mlp', 'gelu')
        self.blocks = nn.ModuleList([
            ValueBlock(
                width, config['heads'], norm, mlp, index in selected,
                config.get('value_initial_scale', 0.0),
            )
            for index in range(config['depth'])
        ])
        self.value_embedding = nn.Embedding(config['vocab'], width)
        self.blocks.apply(GPT.initialize)
        GPT.initialize(self.value_embedding)
        self.freeze_bigram_table = bool(config.get('freeze_bigram_table', False))

    def features(self, ids):
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        token_value = self.value_embedding(ids)
        for block in self.blocks:
            x = block(x, token_value if block.use_value else None)
        return self.norm(x)

    @torch.no_grad()
    def observe_batch(self, inputs, targets):
        if not self.freeze_bigram_table:
            super().observe_batch(inputs, targets)

    @torch.no_grad()
    def finalize_training(self):
        if not self.freeze_bigram_table:
            super().finalize_training()


def build_model(config):
    return ValueUnigramAdaptiveGPT(config)
