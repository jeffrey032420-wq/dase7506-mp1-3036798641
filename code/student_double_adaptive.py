"""Causal GPT with probability-calibrated train n-grams and window copying."""
import torch
from torch.nn import functional as F

from student_copy import recent_copies
from student_ngram import NgramResidualGPT
from student_value_unigram import ValueUnigramAdaptiveGPT


class DoubleAdaptiveGPT(NgramResidualGPT):
    def __init__(self, config):
        super().__init__(config)
        ngram_values = config['adaptive_ngram_bonuses']
        copy_values = config['adaptive_copy_bonuses']
        if len(ngram_values) != 3 or any(len(row) != 3 for row in ngram_values):
            raise ValueError('Expected 3-by-3 n-gram bonus table.')
        if len(copy_values) != 4 or any(len(row) != 3 for row in copy_values):
            raise ValueError('Expected 4-by-3 copy bonus table.')
        self.register_buffer('_ngram_bonus_table', torch.tensor(
            [[0., 0., 0.]] * 2 + ngram_values, dtype=torch.float32), persistent=False)
        self.register_buffer('_copy_bonus_table', torch.tensor(
            [[0., 0., 0.]] + copy_values, dtype=torch.float32), persistent=False)
        self.register_buffer('_probability_edges', torch.tensor(
            [0.03, 0.30], dtype=torch.float32), persistent=False)

    def forward(self, ids):
        logits = ValueUnigramAdaptiveGPT.forward(self, ids)
        batch, length = ids.shape
        ngram_candidate = torch.zeros((batch, length), dtype=torch.long, device=ids.device)
        ngram_order = torch.zeros((batch, length), dtype=torch.long, device=ids.device)
        for order in sorted(self.ngram_sizes):
            if order > length:
                continue
            keys = getattr(self, f'ngram_keys_{order}')
            values = getattr(self, f'ngram_next_{order}')
            query = ids[:, :length-order+1].long().clone()
            for offset in range(1, order):
                query = (query << 11) | ids[:, offset:offset+length-order+1]
            indices = torch.searchsorted(keys, query.contiguous().reshape(-1)).reshape_as(query)
            indices = indices.clamp_max(len(keys)-1)
            found = keys[indices] == query
            ngram_candidate[:, order-1:][found] = values[indices][found]
            ngram_order[:, order-1:][found] = order
        candidate_logit = logits.gather(-1, ngram_candidate.unsqueeze(-1)).squeeze(-1)
        p_ngram = (candidate_logit.float()-torch.logsumexp(logits.float(),dim=-1)).exp()
        ngram_bin = torch.bucketize(p_ngram, self._probability_edges)
        ngram_bonus = self._ngram_bonus_table[ngram_order, ngram_bin].to(logits.dtype)
        logits = logits.scatter_add(-1, ngram_candidate.unsqueeze(-1), ngram_bonus.unsqueeze(-1))
        if length >= 2:
            copy_order, copy_candidate = recent_copies(ids)
            copy_logit = logits.gather(-1, copy_candidate.unsqueeze(-1)).squeeze(-1)
            p_copy = (copy_logit.float()-torch.logsumexp(logits.float(),dim=-1)).exp()
            copy_bin = torch.bucketize(p_copy, self._probability_edges)
            copy_bonus = self._copy_bonus_table[copy_order, copy_bin].to(logits.dtype)
            logits = logits.scatter_add(-1, copy_candidate.unsqueeze(-1), copy_bonus.unsqueeze(-1))
        return logits

    def predict_log_probs(self, ids):
        return F.log_softmax(self.forward(ids).float(),dim=-1)


def build_model(config):
    return DoubleAdaptiveGPT(config)
