"""A18 with a causal empirical distribution over repeated within-window contexts."""
import torch
from torch.nn import functional as F

from student_rare_adaptive import RareAdaptiveGPT


class CopyMixtureGPT(RareAdaptiveGPT):
    def __init__(self, config):
        super().__init__(config)
        alphas = config['copy_mixture_alphas']
        if len(alphas) != 4 or any(not 0 <= float(v) < 1 for v in alphas):
            raise ValueError('Expected four mixture coefficients in [0,1).')
        self.register_buffer('_copy_mixture_alphas',
                             torch.tensor([0.] + list(alphas), dtype=torch.float32),
                             persistent=False)

    def predict_log_probs(self, ids):
        logits = self.forward(ids)
        batch, length = ids.shape
        if length < 2:
            return F.log_softmax(logits.float(), dim=-1)
        positions = torch.arange(length, device=ids.device)
        past = positions[None, :] < positions[:, None]
        equal = torch.ones((batch, length, length), dtype=torch.bool, device=ids.device)
        masks = []
        order = torch.zeros((batch, length), dtype=torch.long, device=ids.device)
        for n in range(1, 5):
            shifted = torch.zeros_like(ids)
            shifted[:, n-1:] = ids[:, :length-n+1]
            equal &= shifted[:, :, None] == shifted[:, None, :]
            allowed = past & (positions[:, None] >= n-1) & (positions[None, :] >= n-1)
            match = equal & allowed
            masks.append(match)
            order[match.any(-1)] = n
        selected = torch.zeros_like(equal)
        for n, match in enumerate(masks, 1):
            selected |= match & (order == n)[:, :, None]
        follower = ids[:, (positions + 1).clamp_max(length-1)]
        follower = follower[:, None, :].expand(batch, length, length)
        counts = torch.zeros_like(logits, dtype=torch.float32)
        counts.scatter_add_(-1, follower, selected.to(torch.float32))
        totals = selected.sum(-1).clamp_min(1).unsqueeze(-1)
        q = counts / totals
        alpha = self._copy_mixture_alphas[order].unsqueeze(-1)
        probabilities = F.softmax(logits.float(), dim=-1)
        probabilities = probabilities * (1 - alpha) + q * alpha
        return probabilities.log()


def build_model(config):
    return CopyMixtureGPT(config)
