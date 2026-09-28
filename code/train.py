"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def _average_keys(state):
    """Return floating tensors that should be averaged, excluding count-derived state."""
    return [key for key, value in state.items()
            if value.is_floating_point() and key != 'bigram.weight']


def _copy_tensors(state, keys):
    return {key: state[key].detach().cpu().clone() for key in keys}


@torch.no_grad()
def _update_ema(average, state, decay):
    for key, value in average.items():
        value.mul_(decay).add_(state[key].detach().cpu(), alpha=1.0 - decay)


@torch.no_grad()
def _update_swa(average, state, keys, count):
    if average is None:
        return _copy_tensors(state, keys)
    coefficient = 1.0 / (count + 1)
    for key, value in average.items():
        value.add_(state[key].detach().cpu() - value, alpha=coefficient)
    return average


def _candidate_checkpoint(model, averaged_state, keys):
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for key in keys:
        state[key].copy_(averaged_state[key])
    return state


def _save_training_state(path, *, args, config, recipe, step, model, optimizer, rng,
                         history, validation_history, ema_states, swa_state, swa_count,
                         train_seconds):
    payload = {
        'protocol': PROTOCOL,
        'implementation': args.implementation,
        'config': config,
        'recipe': recipe,
        'run_dir': str(args.run_dir.resolve()),
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'rng_state': rng.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'history': history,
        'validation_history': validation_history,
        'ema_states': ema_states,
        'swa_state': swa_state,
        'swa_count': swa_count,
        'train_seconds': train_seconds,
    }
    if hasattr(model, '_bigram_counts'):
        payload['bigram_counts'] = model._bigram_counts.detach().cpu().clone()
    torch.save(payload, path)


def _checkpoint_payload(args, config, model_state, train_tokens, averaging='raw',
                        initialized_from=None, source_train_tokens=0):
    payload = {
        'protocol': PROTOCOL,
        'implementation': args.implementation,
        'config': config,
        'model': model_state,
        'seed': args.seed,
        'train_tokens': train_tokens,
        'averaging': averaging,
    }
    if initialized_from is not None:
        payload['initialized_from'] = initialized_from
        payload['source_train_tokens'] = source_train_tokens
        payload['phase_train_tokens'] = train_tokens - source_train_tokens
    return payload


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200,
                   help='Total schedule length, including any resumed steps.')
    p.add_argument('--stop-after-step', type=int, default=0,
                   help='Stop at this completed step and save resumable state; 0 runs to --steps.')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--beta2', type=float, default=.999)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--resume-state', type=Path,
                   help='Resume exactly from a train-state-step-*.pt file.')
    p.add_argument('--init-state', type=Path,
                   help='Start a new optimizer/schedule phase from a training state. '
                        'Model, data RNG, Torch RNG, and count-derived state are continued.')
    p.add_argument('--init-optimizer-state', action='store_true',
                   help='With --init-state, also continue AdamW moments while using the new LR schedule.')
    p.add_argument('--save-every', type=int, default=0,
                   help='Write resumable training state every N completed steps; 0 disables it.')
    p.add_argument('--ema-decays', type=float, nargs='*', default=[],
                   help='Optional EMA decay values, for example 0.999 0.9995.')
    p.add_argument('--swa-start-step', type=int, default=0,
                   help='First completed step sampled into SWA; 0 disables SWA.')
    p.add_argument('--swa-every', type=int, default=0,
                   help='SWA sampling interval; required when SWA is enabled.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.learning_rate <= 0 or not 0 <= args.min_lr_ratio <= 1:
        p.error('Learning rate must be positive and min-lr-ratio must be in [0, 1].')
    if args.warmup_steps < 0 or not 0 < args.beta2 < 1 or args.weight_decay < 0:
        p.error('Invalid optimizer or warmup setting.')
    if args.eval_every < 0 or args.save_every < 0 or args.stop_after_step < 0:
        p.error('Intervals and stop-after-step must be non-negative.')
    if args.init_optimizer_state and args.init_state is None:
        p.error('--init-optimizer-state requires --init-state.')
    if args.stop_after_step > args.steps:
        p.error('--stop-after-step cannot exceed --steps.')
    if len(set(args.ema_decays)) != len(args.ema_decays) or any(
            not 0.0 < decay < 1.0 for decay in args.ema_decays):
        p.error('EMA decays must be unique values strictly between 0 and 1.')
    if (args.swa_start_step == 0) != (args.swa_every == 0):
        p.error('--swa-start-step and --swa-every must both be zero or both be positive.')
    if args.swa_start_step < 0 or args.swa_every < 0 or args.swa_start_step > args.steps:
        p.error('Invalid SWA schedule.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and args.resume_state is None:
        p.error('Run directory already contains results. Use a new --run-dir or --resume-state.')

    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(.9, args.beta2),
        weight_decay=args.weight_decay,
    )
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    initialized_from = None
    source_train_tokens = 0
    source_train_seconds = 0.0
    init_state_sha256 = None
    if args.init_state is not None:
        initial = torch.load(args.init_state, map_location='cpu', weights_only=True)
        if initial.get('protocol') != PROTOCOL:
            p.error('Initialization state belongs to a different course protocol.')
        if initial.get('implementation') != args.implementation:
            p.error('Initialization state uses a different implementation.')
        if initial.get('config') != config:
            p.error('Initialization state config does not match --config.')
        model.load_state_dict(initial['model'])
        if args.init_optimizer_state:
            optimizer.load_state_dict(initial['optimizer'])
        rng.set_state(initial['rng_state'])
        torch.set_rng_state(initial['torch_rng_state'])
        if 'bigram_counts' in initial:
            if not hasattr(model, '_bigram_counts'):
                p.error('Initialization state has bigram counts but the model does not support them.')
            model._bigram_counts.copy_(initial['bigram_counts'].to(device))
        source_recipe = initial.get('recipe', {})
        source_train_tokens = int(
            source_recipe.get('source_train_tokens', 0)
            + initial['step'] * source_recipe['batch_size'] * 256
        )
        source_train_seconds = float(
            source_recipe.get('source_train_seconds', 0.0)
            + initial.get('train_seconds', 0.0)
        )
        initialized_from = str(args.init_state.resolve())
        init_state_sha256 = sha(args.init_state)
    recipe = {
        'implementation': args.implementation,
        'config': config,
        'seed': args.seed,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'precision': precision,
        'learning_rate': args.learning_rate,
        'min_lr_ratio': args.min_lr_ratio,
        'warmup_steps': args.warmup_steps,
        'weight_decay': args.weight_decay,
        'beta2': args.beta2,
        'eval_every': args.eval_every,
        'ema_decays': args.ema_decays,
        'swa_start_step': args.swa_start_step,
        'swa_every': args.swa_every,
        'initialized_from': initialized_from,
        'init_state_sha256': init_state_sha256,
        'source_train_tokens': source_train_tokens,
        'source_train_seconds': source_train_seconds,
        'init_optimizer_state': args.init_optimizer_state,
    }

    state = model.state_dict()
    average_keys = _average_keys(state)
    ema_states = {str(decay): _copy_tensors(state, average_keys)
                  for decay in args.ema_decays}
    swa_state, swa_count = None, 0
    history, validation_history = [], []
    start_step, prior_train_seconds = 0, 0.0
    resumed_from = None
    if args.resume_state is not None:
        resume = torch.load(args.resume_state, map_location='cpu', weights_only=True)
        if resume.get('protocol') != PROTOCOL:
            p.error('Resume state belongs to a different course protocol.')
        if resume.get('recipe') != recipe:
            p.error('Resume state recipe does not match the current training arguments.')
        if Path(resume.get('run_dir', '')).resolve() != args.run_dir.resolve():
            p.error('Resume state belongs to a different --run-dir.')
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        rng.set_state(resume['rng_state'])
        torch.set_rng_state(resume['torch_rng_state'])
        if 'bigram_counts' in resume:
            if not hasattr(model, '_bigram_counts'):
                p.error('Resume state has bigram counts but the model does not support them.')
            model._bigram_counts.copy_(resume['bigram_counts'].to(device))
        ema_states = resume['ema_states']
        swa_state, swa_count = resume['swa_state'], resume['swa_count']
        history = resume['history']
        validation_history = resume['validation_history']
        start_step = resume['step']
        prior_train_seconds = float(resume.get('train_seconds', 0.0))
        resumed_from = str(args.resume_state.resolve())
        if start_step >= args.steps:
            p.error('Resume state has already reached the requested total step count.')

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    intermediate_validation_seconds = 0.0
    end_step = args.stop_after_step or args.steps
    if end_step <= start_step:
        p.error('--stop-after-step must be greater than the resumed step.')

    for step in range(start_step, end_step):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        observer = getattr(model, 'observe_batch', None)
        if observer is not None:
            observer(batch[:,:-1], batch[:,1:])
        warmup = 1. if args.warmup_steps == 0 else min(1., (step + 1) / args.warmup_steps)
        cosine = .5 * (1 + math.cos(math.pi * step / args.steps))
        learning_rate = args.learning_rate * warmup * (
            args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine
        )
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        completed_step = step + 1
        current_state = model.state_dict()
        for decay in args.ema_decays:
            _update_ema(ema_states[str(decay)], current_state, decay)
        if (args.swa_start_step > 0 and completed_step >= args.swa_start_step
                and (completed_step - args.swa_start_step) % args.swa_every == 0):
            swa_state = _update_swa(swa_state, current_state, average_keys, swa_count)
            swa_count += 1
        if completed_step%100 == 0 or completed_step == end_step:
            row = {
                'step': completed_step,
                'loss': loss.item(),
                'seconds': prior_train_seconds + time.perf_counter()-started-intermediate_validation_seconds,
            }
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and completed_step%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':completed_step,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
        if args.save_every > 0 and completed_step%args.save_every == 0:
            elapsed = prior_train_seconds + time.perf_counter()-started-intermediate_validation_seconds
            _save_training_state(
                args.run_dir/f'train-state-step-{completed_step}.pt', args=args,
                config=config, recipe=recipe, step=completed_step, model=model,
                optimizer=optimizer, rng=rng, history=history,
                validation_history=validation_history, ema_states=ema_states,
                swa_state=swa_state, swa_count=swa_count, train_seconds=elapsed,
            )

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = prior_train_seconds + time.perf_counter()-started-intermediate_validation_seconds
    if end_step < args.steps:
        state_path = args.run_dir/f'train-state-step-{end_step}.pt'
        _save_training_state(
            state_path, args=args, config=config, recipe=recipe, step=end_step,
            model=model, optimizer=optimizer, rng=rng, history=history,
            validation_history=validation_history, ema_states=ema_states,
            swa_state=swa_state, swa_count=swa_count, train_seconds=train_seconds,
        )
        print(json.dumps({
            'status': 'paused',
            'step': end_step,
            'total_steps': args.steps,
            'train_state': str(state_path),
            'train_seconds': train_seconds,
        }, indent=2), flush=True)
        return

    finalizer = getattr(model, 'finalize_training', None)
    if finalizer is not None:
        finalizer()
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    phase_train_tokens = args.steps*args.batch_size*256
    train_tokens = source_train_tokens + phase_train_tokens
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save(_checkpoint_payload(
        args, config, {key: value.detach().cpu() for key, value in model.state_dict().items()},
        train_tokens, initialized_from=initialized_from,
        source_train_tokens=source_train_tokens), checkpoint)
    candidates = {'raw': {'path': str(checkpoint), 'sha256': sha(checkpoint)}}
    for decay in args.ema_decays:
        label = f'ema-{decay}'
        candidate = args.run_dir/f'checkpoint-{label}.pt'
        torch.save(_checkpoint_payload(
            args, config, _candidate_checkpoint(model, ema_states[str(decay)], average_keys),
            train_tokens, label, initialized_from, source_train_tokens), candidate)
        candidates[label] = {'path': str(candidate), 'sha256': sha(candidate)}
    if swa_state is not None:
        candidate = args.run_dir/'checkpoint-swa.pt'
        torch.save(_checkpoint_payload(
            args, config, _candidate_checkpoint(model, swa_state, average_keys),
            train_tokens, f'swa-{swa_count}-samples', initialized_from,
            source_train_tokens), candidate)
        candidates['swa'] = {'path': str(candidate), 'sha256': sha(candidate),
                             'samples': swa_count}

    result = {
        'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
        'parameters':sum(parameter.numel() for parameter in model.parameters()),'precision':precision,
        'train_tokens':train_tokens,'phase_train_tokens':phase_train_tokens,
        'source_train_tokens':source_train_tokens,
        'source_train_seconds':source_train_seconds,
        'initialized_from':initialized_from,
        'init_state_sha256':init_state_sha256,
        'preparation_seconds':preparation_seconds,
        'train_seconds':train_seconds,'validation':validation,'history':history,
        'validation_history':validation_history,
        'intermediate_validation_seconds':intermediate_validation_seconds,
        'optimizer':{'name':'AdamW','learning_rate':args.learning_rate,
                     'betas':[.9,args.beta2],'weight_decay':args.weight_decay,
                     'warmup_steps':args.warmup_steps,
                     'min_lr_ratio':args.min_lr_ratio},
        'averaging':{'ema_decays':args.ema_decays,'swa_start_step':args.swa_start_step,
                     'swa_every':args.swa_every,'swa_samples':swa_count},
        'resumed_from':resumed_from,
        'candidates':candidates,
        'process_seconds':time.perf_counter()-total_started,
        'torch_version':str(torch.__version__),'threads':args.threads,
        'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
        **device_metrics(device),
    }
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
