"""Read and verify only the supplied training text and fixed tokenizer."""
import json

import numpy as np
import torch
from tokenizers import Tokenizer

from common import ROOT,sha


def train_ids_numpy():
    directory=ROOT/'data'
    manifest=json.loads((directory/'manifest.json').read_text())
    for name in ('tokenizer.json','wikitext_train.txt'):
        if sha(directory/name)!=manifest['sha256'][name]:
            raise ValueError(f'Changed supplied training asset: {name}')
    tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
    content=(directory/'wikitext_train.txt').read_text(encoding='utf-8')
    return np.asarray(tokenizer.encode(content).ids,dtype=np.uint64)


def train_ids_torch():
    return torch.from_numpy(train_ids_numpy().astype(np.int64))
