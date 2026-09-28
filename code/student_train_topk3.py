"""A20 plus a compact train-derived trigram-context probability table."""
import torch

from student_copy_mixture import CopyMixtureGPT
from student_train_topk import TrainTopKGPT


class TrainTopK3GPT(TrainTopKGPT):
    def __init__(self, config):
        super().__init__(config)
        size=config['train_topk3_size']
        self.train_topk3_alpha=float(config['train_topk3_alpha'])
        if size<1 or not 0<=self.train_topk3_alpha<1:
            raise ValueError('Invalid third-order training table.')
        self.register_buffer('train_topk3_keys',torch.zeros(size,dtype=torch.long))
        self.register_buffer('train_topk3_next',torch.zeros((size,4),dtype=torch.int16))
        self.register_buffer('train_topk3_prob',torch.zeros((size,4),dtype=torch.float16))

    def _apply_train_topk3(self,probabilities,ids):
        batch,length=ids.shape
        if length<3:
            return probabilities
        query=((ids[:,:-2].long()<<11)|ids[:,1:-1].long())<<11|ids[:,2:].long()
        indices=torch.searchsorted(self.train_topk3_keys,query.contiguous().reshape(-1))
        indices=indices.reshape_as(query).clamp_max(len(self.train_topk3_keys)-1)
        found=self.train_topk3_keys[indices]==query
        candidates=self.train_topk3_next[indices].long()
        weights=self.train_topk3_prob[indices].float()*found.unsqueeze(-1)
        values=torch.zeros((batch,length,4),dtype=torch.float32,device=ids.device)
        tokens=torch.zeros((batch,length,4),dtype=torch.long,device=ids.device)
        values[:,2:]=weights
        tokens[:,2:]=candidates
        probabilities*=(1.-self.train_topk3_alpha*values.sum(-1,keepdim=True))
        probabilities.scatter_add_(-1,tokens,self.train_topk3_alpha*values)
        return probabilities

    def predict_log_probs(self,ids):
        probabilities=CopyMixtureGPT.predict_log_probs(self,ids).exp()
        probabilities=self._apply_train_topk(probabilities,ids)
        return self._apply_train_topk3(probabilities,ids).log()


def build_model(config):
    return TrainTopK3GPT(config)
