"""Train a larger neural backbone using only supplied training text and static train-derived memory."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, make_model, setup, sha
from train_only_data import train_ids_torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-source', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--peak-lr', type=float, default=0.0007)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=27)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Run directory must be new or empty.')
    device, _ = setup('cuda', 'fp32', 4)
    source = torch.load(args.memory_source, map_location='cpu', weights_only=True)
    if source['protocol'] != PROTOCOL or source['implementation'] != 'student_copy_adaptive':
        parser.error('Expected A15 static-memory source.')
    config = copy.deepcopy(source['config'])
    config['width'] = args.width
    config['depth'] = args.depth
    config['value_layers'] = [3, 4, 5]
    torch.manual_seed(args.seed)
    model, _ = make_model('student_copy_adaptive', config, device)
    with torch.no_grad():
        model.bigram.weight.copy_(source['model']['bigram.weight'].to(device))
        for order in (2, 3, 4):
            getattr(model, f'ngram_keys_{order}').copy_(
                source['model'][f'ngram_keys_{order}'].to(device))
            getattr(model, f'ngram_next_{order}').copy_(
                source['model'][f'ngram_next_{order}'].to(device))
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    moments = [(torch.zeros_like(p), torch.zeros_like(p)) for p in parameters]
    train = train_ids_torch().to(device)
    offsets = torch.arange(257, device=device)
    generator = torch.Generator().manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        starts = torch.randint(len(train)-257, (args.batch_size,), generator=generator).to(device)
        batch = train[starts[:, None]+offsets]
        for parameter in parameters:
            parameter.grad = None
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        warmup = min(1., step/200)
        cosine = .5*(1+math.cos(math.pi*step/args.steps))
        lr = args.peak_lr*warmup*(.01+.99*cosine)
        with torch.no_grad():
            for parameter, (first, second) in zip(parameters, moments):
                if parameter.grad is None:
                    continue
                first.mul_(.9).add_(parameter.grad, alpha=.1)
                second.mul_(.95).addcmul_(parameter.grad, parameter.grad, value=.05)
                corrected_first = first/(1-.9**step)
                corrected_second = second/(1-.95**step)
                parameter.mul_(1-lr*.1)
                parameter.addcdiv_(corrected_first, corrected_second.sqrt().add_(1e-8), value=-lr)
        if step == 1 or step % 500 == 0:
            torch.cuda.synchronize()
            print(json.dumps({'step': step, 'train_loss': float(loss.detach()),
                              'lr': lr, 'seconds': time.perf_counter()-start,
                              'gpu_peak_gib': torch.cuda.max_memory_allocated()/2**30}),
                  flush=True)
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = {'protocol': PROTOCOL, 'implementation': 'student_copy_adaptive',
                          'config': config, 'model': {key: value.detach().cpu().clone()
                                                    for key,value in model.state_dict().items()},
                          'seed': args.seed, 'train_tokens': step*args.batch_size*256,
                          'memory_source_sha256': sha(args.memory_source),
                          'memory_source_train_tokens': source['train_tokens'],
                          'ngram_training_tokens': source['ngram_training_tokens'],
                          'recipe': {'width': args.width, 'depth': args.depth,
                                     'steps_completed': step, 'batch_size': args.batch_size,
                                     'peak_lr': args.peak_lr, 'seed': args.seed,
                                     'seconds': time.perf_counter()-start}}
            output = args.run_dir / f'checkpoint-step-{step}.pt'
            torch.save(checkpoint, output)
            print(json.dumps({'saved': str(output), 'bytes': output.stat().st_size}),flush=True)


if __name__ == '__main__':
    main()
