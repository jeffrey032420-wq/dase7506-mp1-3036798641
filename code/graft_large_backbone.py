"""Combine a separately trained wider backbone with unchanged train-only A21 tables."""
import argparse
import copy
import json
from pathlib import Path

import torch

from common import PROTOCOL,make_model,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-backbone',required=True,type=Path)
    parser.add_argument('--source-memory',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    backbone=torch.load(args.source_backbone,map_location='cpu',weights_only=True)
    memory=torch.load(args.source_memory,map_location='cpu',weights_only=True)
    if (backbone['protocol']!=PROTOCOL or memory['protocol']!=PROTOCOL or
            backbone['implementation'] not in ('student_copy_adaptive','student_copy_dropout') or
            memory['implementation']!='student_train_topk3'):
        parser.error('Expected large A15 backbone and A21 memory.')
    result=copy.deepcopy(memory)
    for name in ('width','depth','heads','value_layers'):
        result['config'][name]=copy.deepcopy(backbone['config'][name])
    model,_=make_model(result['implementation'],result['config'],torch.device('cpu'))
    state=model.state_dict()
    for name,value in state.items():
        if name in backbone['model'] and backbone['model'][name].shape==value.shape:
            state[name]=backbone['model'][name].clone()
        elif name in memory['model'] and memory['model'][name].shape==value.shape:
            state[name]=memory['model'][name].clone()
        else:
            parser.error(f'Missing tensor with correct shape: {name}')
    result['model']=state
    result['train_tokens']=backbone['train_tokens']
    result['backbone_sha256']=sha(args.source_backbone)
    result['memory_sha256']=sha(args.source_memory)
    result['large_recipe']=backbone['recipe']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.output)
    print(json.dumps({'output':str(args.output),'width':result['config']['width'],
                      'depth':result['config']['depth'],'train_tokens':result['train_tokens'],
                      'bytes':args.output.stat().st_size}))


if __name__=='__main__':
    main()
