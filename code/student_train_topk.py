"""A19 with a compact training-only top-four conditional distribution."""
import torch

from student_copy_mixture import CopyMixtureGPT


class TrainTopKGPT(CopyMixtureGPT):
    def __init__(self, config):
        super().__init__(config)
        size = config['train_topk_size']
        self.train_topk_alpha = float(config['train_topk_alpha'])
        if size < 1 or not 0 <= self.train_topk_alpha < 1:
            raise ValueError('Invalid training conditional table.')
        self.register_buffer('train_topk_keys', torch.zeros(size, dtype=torch.long))
        self.register_buffer('train_topk_next', torch.zeros((size, 4), dtype=torch.int16))
        self.register_buffer('train_topk_prob', torch.zeros((size, 4), dtype=torch.float16))

    def _apply_train_topk(self, probabilities, ids):
        batch, length = ids.shape
        if length < 2:
            return probabilities
        query = (ids[:, :-1].long() << 11) | ids[:, 1:].long()
        indices = torch.searchsorted(self.train_topk_keys, query.contiguous().reshape(-1))
        indices = indices.reshape_as(query).clamp_max(len(self.train_topk_keys)-1)
        found = self.train_topk_keys[indices] == query
        candidates = self.train_topk_next[indices].long()
        weights = self.train_topk_prob[indices].float() * found.unsqueeze(-1)
        values = torch.zeros((batch, length, 4), dtype=torch.float32, device=ids.device)
        tokens = torch.zeros((batch, length, 4), dtype=torch.long, device=ids.device)
        values[:, 1:] = weights
        tokens[:, 1:] = candidates
        probabilities *= (1. - self.train_topk_alpha * values.sum(-1, keepdim=True))
        probabilities.scatter_add_(-1, tokens, self.train_topk_alpha * values)
        return probabilities

    def predict_log_probs(self, ids):
        probabilities = super().predict_log_probs(ids).exp()
        return self._apply_train_topk(probabilities, ids).log()


def build_model(config):
    return TrainTopKGPT(config)
