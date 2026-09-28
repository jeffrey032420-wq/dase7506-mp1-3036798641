"""Attach train-only top-four trigram conditionals for contexts seen at least twice."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from analyze_ngram_memory import pair_keys
from common import PROTOCOL,sha
from train_only_data import train_ids_numpy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--alpha',type=float,default=.1)
    parser.add_argument('--min-count',type=int,default=2)
    args=parser.parse_args()
    source=torch.load(args.source,map_location='cpu',weights_only=True)
    if source['protocol']!=PROTOCOL or source['implementation']!='student_train_topk':
        parser.error('Expected A20 source.')
    train=train_ids_numpy()
    pairs,counts=np.unique(pair_keys(train,3),return_counts=True)
    contexts=pairs>>11
    first=np.r_[0,np.flatnonzero(contexts[1:]!=contexts[:-1])+1]
    all_keys=contexts[first]
    totals=np.add.reduceat(counts,first)
    selected=totals>=args.min_count
    keys=all_keys[selected]
    ranked=np.lexsort((pairs&2047,-counts,contexts))
    ranked_context=contexts[ranked]
    ranked_first=np.r_[0,np.flatnonzero(ranked_context[1:]!=ranked_context[:-1])+1][selected]
    next_ids=np.zeros((len(keys),4),dtype=np.int16)
    probabilities=np.zeros((len(keys),4),dtype=np.float16)
    for offset in range(4):
        index=ranked_first+offset
        valid=index<len(ranked)
        index=np.minimum(index,len(ranked)-1)
        valid &= ranked_context[index]==keys
        next_ids[valid,offset]=(pairs[ranked[index[valid]]]&2047).astype(np.int16)
        probabilities[valid,offset]=(counts[ranked[index[valid]]]/totals[selected][valid]).astype(np.float16)
    result=copy.deepcopy(source)
    result['implementation']='student_train_topk3'
    result['config']['train_topk3_size']=len(keys)
    result['config']['train_topk3_alpha']=args.alpha
    result['config']['train_topk3_min_count']=args.min_count
    result['model']['train_topk3_keys']=torch.from_numpy(keys.astype(np.int64).copy())
    result['model']['train_topk3_next']=torch.from_numpy(next_ids)
    result['model']['train_topk3_prob']=torch.from_numpy(probabilities)
    result['initialized_from_sha256']=sha(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.output)
    print(json.dumps({'contexts':len(keys),'alpha':args.alpha,'min_count':args.min_count,
                      'bytes':args.output.stat().st_size,'output':str(args.output)},indent=2))


if __name__=='__main__':
    main()
