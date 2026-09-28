"""A17 unique training 4-grams with confidence-calibrated logit bonuses."""
import torch
from torch.nn import functional as F

from student_double_adaptive import DoubleAdaptiveGPT
from student_rare_ngram import RareNgramGPT


class RareAdaptiveGPT(RareNgramGPT):
    def __init__(self, config):
        super().__init__(config)
        values=config['adaptive_rare_bonuses']
        if len(values)!=3 or any(float(value)<0 for value in values):
            raise ValueError('Expected three nonnegative rare n-gram bonuses.')
        self.register_buffer('_rare_bonus_table',torch.tensor(values,dtype=torch.float32),persistent=False)
        self.register_buffer('_rare_probability_edges',torch.tensor([.03,.30],dtype=torch.float32),persistent=False)

    def forward(self, ids):
        logits=DoubleAdaptiveGPT.forward(self,ids)
        length=ids.shape[1]
        if length<4:
            return logits
        query=ids[:,:length-3].long().clone()
        for offset in (1,2,3):
            query=(query<<11)|ids[:,offset:offset+length-3]
        indices=torch.searchsorted(self.rare_ngram_keys,
                                   query.contiguous().reshape(-1)).reshape_as(query)
        indices=indices.clamp_max(len(self.rare_ngram_keys)-1)
        found=self.rare_ngram_keys[indices]==query
        candidate=torch.zeros_like(ids)
        candidate[:,3:]=self.rare_ngram_next[indices].long()
        candidate_logit=logits.gather(-1,candidate.unsqueeze(-1)).squeeze(-1)
        probability=(candidate_logit.float()-torch.logsumexp(logits.float(),dim=-1)).exp()
        bins=torch.bucketize(probability,self._rare_probability_edges)
        bonuses=self._rare_bonus_table[bins].to(logits.dtype)
        bonuses[:,:3]=0
        bonuses[:,3:]=bonuses[:,3:]*found
        return logits.scatter_add(-1,candidate.unsqueeze(-1),bonuses.unsqueeze(-1))

    def predict_log_probs(self,ids):
        return F.log_softmax(self.forward(ids).float(),dim=-1)


def build_model(config):
    return RareAdaptiveGPT(config)
