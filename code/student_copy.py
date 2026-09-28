"""A13 train-text n-gram model with strictly within-window continuation copying."""
import torch

from student_ngram import NgramResidualGPT


def recent_copies(ids, max_order=4):
    """Return the longest previously seen suffix and its already observed follower."""
    batch, length = ids.shape
    positions = torch.arange(length, device=ids.device)
    past = positions[None, :] < positions[:, None]
    candidate = torch.zeros_like(ids)
    longest = torch.zeros_like(ids)
    equal = torch.ones((batch, length, length), dtype=torch.bool, device=ids.device)
    for order in range(1, max_order + 1):
        shifted = torch.zeros_like(ids)
        shifted[:, order-1:] = ids[:, :length-order+1]
        equal &= shifted[:, :, None] == shifted[:, None, :]
        allowed = past & (positions[:, None] >= order-1) & (positions[None, :] >= order-1)
        prior = torch.where(equal & allowed, positions[None, None, :], -1).amax(-1)
        found = prior >= 0
        next_token = ids.gather(1, (prior+1).clamp_min(0))
        candidate[found] = next_token[found]
        longest[found] = order
    return longest, candidate


class CopyNgramGPT(NgramResidualGPT):
    def __init__(self, config):
        super().__init__(config)
        self.copy_bonuses = [0.0] + [float(v) for v in config['copy_bonuses']]
        if len(self.copy_bonuses) != 5 or any(v < 0 for v in self.copy_bonuses):
            raise ValueError('Expected four non-negative copy bonuses.')

    def forward(self, ids):
        logits = super().forward(ids)
        if ids.shape[1] < 2:
            return logits
        order, candidate = recent_copies(ids)
        weights = torch.tensor(self.copy_bonuses, dtype=logits.dtype, device=ids.device)
        bonus = weights[order]
        return logits.scatter_add(-1, candidate.unsqueeze(-1), bonus.unsqueeze(-1))


def build_model(config):
    return CopyNgramGPT(config)
