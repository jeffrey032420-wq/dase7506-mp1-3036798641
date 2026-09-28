"""A12 causal GPT with a compact train-text n-gram next-token residual."""
import torch
from torch.nn import functional as F

from student_value_unigram import ValueUnigramAdaptiveGPT


class NgramResidualGPT(ValueUnigramAdaptiveGPT):
    """Boost a train-derived continuation matching the observed within-window suffix.

    Every lookup uses tokens at or before the scored position. Temporary lookup
    results are local to each forward call and to each batch row.
    """

    def __init__(self, config):
        super().__init__(config)
        self.ngram_bonuses = {int(k): float(v) for k, v in
                              config.get('ngram_bonuses', {}).items()}
        self.ngram_sizes = {int(k): int(v) for k, v in
                            config.get('ngram_sizes', {}).items()}
        if set(self.ngram_bonuses) != set(self.ngram_sizes):
            raise ValueError('N-gram bonus and table orders must match.')
        for order, size in self.ngram_sizes.items():
            if order < 2 or order > 5 or size < 1 or self.ngram_bonuses[order] < 0:
                raise ValueError('Invalid n-gram order, table size or bonus.')
            self.register_buffer(f'ngram_keys_{order}', torch.zeros(size, dtype=torch.long))
            self.register_buffer(f'ngram_next_{order}', torch.zeros(size, dtype=torch.long))

    def forward(self, ids):
        logits = super().forward(ids)
        batch, length = ids.shape
        if not self.ngram_sizes:
            return logits
        candidate = torch.zeros((batch, length), dtype=torch.long, device=ids.device)
        bonus = torch.zeros((batch, length), dtype=logits.dtype, device=ids.device)
        for order in sorted(self.ngram_sizes):
            if order > length:
                continue
            keys = getattr(self, f'ngram_keys_{order}')
            next_ids = getattr(self, f'ngram_next_{order}')
            query = ids[:, :length-order+1].long().clone()
            for offset in range(1, order):
                query = (query << 11) | ids[:, offset:offset+length-order+1]
            indices = torch.searchsorted(keys, query.contiguous().reshape(-1)).reshape_as(query)
            indices = indices.clamp_max(len(keys)-1)
            found = keys[indices] == query
            row = candidate[:, order-1:]
            row[found] = next_ids[indices][found]
            weight = bonus[:, order-1:]
            weight[found] = self.ngram_bonuses[order]
        return logits.scatter_add(-1, candidate.unsqueeze(-1), bonus.unsqueeze(-1))

    def predict_log_probs(self, ids):
        return F.log_softmax(self.forward(ids).float(), dim=-1)


def build_model(config):
    return NgramResidualGPT(config)
