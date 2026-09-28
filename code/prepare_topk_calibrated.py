"""Set validation-selected strengths of fixed train-derived probability tables."""
import argparse
import copy
import json
from pathlib import Path

import torch

from common import PROTOCOL,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--alpha2',required=True,type=float)
    parser.add_argument('--alpha3',required=True,type=float)
    args=parser.parse_args()
    source=torch.load(args.source,map_location='cpu',weights_only=True)
    if source['protocol']!=PROTOCOL or source['implementation']!='student_train_topk3':
        parser.error('Expected A21 checkpoint.')
    if not 0<=args.alpha2<1 or not 0<=args.alpha3<1:
        parser.error('Alpha must lie in [0,1).')
    result=copy.deepcopy(source)
    result['config']['train_topk_alpha']=args.alpha2
    result['config']['train_topk3_alpha']=args.alpha3
    result['initialized_from_sha256']=sha(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.output)
    print(json.dumps({'output':str(args.output),'alpha2':args.alpha2,
                      'alpha3':args.alpha3,'bytes':args.output.stat().st_size}))


if __name__=='__main__':
    main()
