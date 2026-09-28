"""Derive a compact top-four bigram-context probability table from train only."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from analyze_ngram_memory import pair_keys
from common import PROTOCOL, sha
from train_only_data import train_ids_numpy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--alpha',type=float,default=.15)
    args=parser.parse_args()
    source=torch.load(args.source,map_location='cpu',weights_only=True)
    if source['protocol']!=PROTOCOL or source['implementation']!='student_copy_mixture':
        parser.error('Expected A19 source.')
    train=train_ids_numpy()
    pairs,counts=np.unique(pair_keys(train,2),return_counts=True)
    contexts=pairs>>11
    first=np.r_[0,np.flatnonzero(contexts[1:]!=contexts[:-1])+1]
    keys=contexts[first]
    total=np.add.reduceat(counts,first)
    ranked=np.lexsort((pairs&2047,-counts,contexts))
    ranked_context=contexts[ranked]
    ranked_first=np.r_[0,np.flatnonzero(ranked_context[1:]!=ranked_context[:-1])+1]
    next_ids=np.zeros((len(keys),4),dtype=np.int16)
    probabilities=np.zeros((len(keys),4),dtype=np.float16)
    for offset in range(4):
        index=ranked_first+offset
        valid=index<len(ranked)
        index=np.minimum(index,len(ranked)-1)
        valid &= ranked_context[index]==keys
        next_ids[valid,offset]=(pairs[ranked[index[valid]]]&2047).astype(np.int16)
        probabilities[valid,offset]=(counts[ranked[index[valid]]]/total[valid]).astype(np.float16)
    result=copy.deepcopy(source)
    result['implementation']='student_train_topk'
    result['config']['train_topk_size']=len(keys)
    result['config']['train_topk_alpha']=args.alpha
    result['model']['train_topk_keys']=torch.from_numpy(keys.astype(np.int64).copy())
    result['model']['train_topk_next']=torch.from_numpy(next_ids)
    result['model']['train_topk_prob']=torch.from_numpy(probabilities)
    result['initialized_from_sha256']=sha(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.output)
    print(json.dumps({'contexts':len(keys),'alpha':args.alpha,'bytes':args.output.stat().st_size,
                      'output':str(args.output)},indent=2))


if __name__=='__main__':
    main()
