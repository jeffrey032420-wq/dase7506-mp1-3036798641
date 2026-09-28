"""GPT with a learned token-to-next-token residual.

The small baseline must route even very local lexical regularities through its
limited transformer width.  This variant gives it a direct bigram path: the row
indexed by the currently observed token is added to the transformer's next-token
logits.  The table is derived from token pairs observed in the ordinary training
batches, so it uses no information outside the supplied training split.  It is
strictly causal and adds one inexpensive embedding lookup at inference time.

Set ``bigram_enabled`` to false for the mechanism ablation.
"""
import torch
from torch import nn

from model import GPT


class BigramResidualGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
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
        """Accumulate exactly the target pairs already processed by the trainer."""
        if not self.bigram_enabled:
            return
        valid = targets >= 0
        pairs = inputs[valid] * self.config['vocab'] + targets[valid]
        flat_counts = self._bigram_counts.view(-1)
        flat_counts.index_add_(0, pairs, torch.ones_like(pairs, dtype=flat_counts.dtype))

    @torch.no_grad()
    def finalize_training(self):
        """Convert observed counts to a smoothed conditional/unigram log ratio."""
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
        residual = (conditional.log() - unigram.log()).clamp(-self.bigram_clip, self.bigram_clip)
        self.bigram.weight.copy_(residual)

    def forward(self, ids):
        logits = super().forward(ids)
        if self.bigram_enabled:
            logits = logits + self.bigram_scale * self.bigram(ids)
        return logits


def build_model(config):
    return BigramResidualGPT(config)
