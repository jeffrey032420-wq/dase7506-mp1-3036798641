"""Frequency-adaptive A8 predictor with a training-unigram output bias."""
import torch

from student_adaptive import AdaptiveBigramResidualGPT


class UnigramAdaptiveGPT(AdaptiveBigramResidualGPT):
    def __init__(self, config):
        super().__init__(config)
        values = config.get('unigram_log_bias')
        if values is None or len(values) != config['vocab']:
            raise ValueError('unigram_log_bias must contain one value per token.')
        self.unigram_scale = float(config.get('unigram_scale', 0.0))
        self.register_buffer(
            '_unigram_log_bias', torch.tensor(values, dtype=torch.float32), persistent=False
        )

    def forward(self, ids):
        return super().forward(ids) + self.unigram_scale * self._unigram_log_bias


def build_model(config):
    return UnigramAdaptiveGPT(config)
