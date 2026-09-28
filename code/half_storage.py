"""Store learned floating-point weights in half precision; load into FP32 CPU modules."""
import argparse
import copy
import json
from pathlib import Path

import torch

from common import PROTOCOL, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    source=torch.load(args.source,map_location='cpu',weights_only=True)
    if source['protocol']!=PROTOCOL:
        parser.error('Wrong protocol.')
    result=copy.deepcopy(source)
    changed=[]
    for key,value in result['model'].items():
        if value.dtype==torch.float32 and value.numel()>0:
            result['model'][key]=value.to(torch.float16)
            changed.append(key)
    result['storage_precision']='fp16; model loads into CPU FP32'
    result['storage_source_sha256']=sha(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.output)
    print(json.dumps({'output':str(args.output),'bytes':args.output.stat().st_size,
                      'converted_tensors':len(changed)}))


if __name__=='__main__':
    main()
