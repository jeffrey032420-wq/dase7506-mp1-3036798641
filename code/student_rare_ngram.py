"""A16 with a compact table of unique training-text four-token continuations."""
import torch
from torch.nn import functional as F

from student_double_adaptive import DoubleAdaptiveGPT


class RareNgramGPT(DoubleAdaptiveGPT):
    def __init__(self, config):
        super().__init__(config)
        size = int(config['rare_ngram_size'])
        if size <= 0 or float(config['rare_ngram_bonus']) < 0:
            raise ValueError('Invalid rare n-gram table size or bonus.')
        self.rare_ngram_bonus = float(config['rare_ngram_bonus'])
        self.register_buffer('rare_ngram_keys', torch.zeros(size, dtype=torch.long))
        self.register_buffer('rare_ngram_next', torch.zeros(size, dtype=torch.int16))

    def forward(self, ids):
        logits = super().forward(ids)
        length = ids.shape[1]
        if length < 4:
            return logits
        query = ids[:, :length-3].long().clone()
        for offset in (1, 2, 3):
            query = (query << 11) | ids[:, offset:offset+length-3]
        indices = torch.searchsorted(self.rare_ngram_keys,
                                     query.contiguous().reshape(-1)).reshape_as(query)
        indices = indices.clamp_max(len(self.rare_ngram_keys)-1)
        found = self.rare_ngram_keys[indices] == query
        candidate = torch.zeros_like(ids)
        bonus = torch.zeros(ids.shape, dtype=logits.dtype, device=ids.device)
        candidate[:, 3:][found] = self.rare_ngram_next[indices][found].long()
        bonus[:, 3:][found] = self.rare_ngram_bonus
        return logits.scatter_add(-1, candidate.unsqueeze(-1), bonus.unsqueeze(-1))

    def predict_log_probs(self, ids):
        return F.log_softmax(self.forward(ids).float(), dim=-1)


def build_model(config):
    return RareNgramGPT(config)
