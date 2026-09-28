"""A7 transformer with frequency-adaptive training-count residual scaling."""
import torch

from model import GPT
from student import BigramResidualGPT


class AdaptiveBigramResidualGPT(BigramResidualGPT):
    """Use validation-selected temperature and per-token residual scales.

    The token-scale vector is fully specified by the checkpoint configuration.
    It is selected from bins defined only by training-set bigram row counts.
    """

    def __init__(self, config):
        super().__init__(config)
        self.logit_temperature = float(config.get('logit_temperature', 1.0))
        if self.logit_temperature <= 0:
            raise ValueError('logit_temperature must be positive.')
        scales = config.get('bigram_token_scales')
        if scales is None:
            scales = [self.bigram_scale] * config['vocab']
        if len(scales) != config['vocab'] or any(float(scale) < 0 for scale in scales):
            raise ValueError('bigram_token_scales must contain one non-negative value per token.')
        self.register_buffer(
            '_bigram_token_scales',
            torch.tensor(scales, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, ids):
        logits = GPT.forward(self, ids) / self.logit_temperature
        if self.bigram_enabled:
            scales = self._bigram_token_scales[ids].unsqueeze(-1)
            logits = logits + scales * self.bigram(ids)
        return logits


def build_model(config):
    return AdaptiveBigramResidualGPT(config)
