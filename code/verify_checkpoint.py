"""Run the prediction contract checks against an exact serialized checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from common import PROTOCOL, make_model, setup, sha


def maximum_difference(left, right):
    return float((left - right).abs().max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('protocol') != PROTOCOL:
        parser.error('Checkpoint belongs to a different protocol.')
    model, implementation_sha = make_model(
        checkpoint['implementation'], checkpoint['config'], device
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()
    torch.manual_seed(7506)
    x = torch.randint(0, checkpoint['config']['vocab'], (2, 12), device=device)
    changed = x.clone()
    changed[:, 7:] = (changed[:, 7:] + 19) % checkpoint['config']['vocab']
    with torch.no_grad():
        first = model.predict_log_probs(x)
        future_changed = model.predict_log_probs(changed)
        alone = model.predict_log_probs(x[:1])
        model.predict_log_probs((x + 31) % checkpoint['config']['vocab'])
        repeated = model.predict_log_probs(x)
    model.zero_grad(set_to_none=True)
    training_logits = model(x)
    training_loss = F.cross_entropy(training_logits[:, :-1].flatten(0, 1),
                                    x[:, 1:].flatten())
    training_loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    result = {
        'protocol': PROTOCOL,
        'checkpoint_sha256': sha(args.checkpoint),
        'implementation_sha256': implementation_sha,
        'finite': bool(torch.isfinite(first).all()),
        'causality_max_abs': maximum_difference(first[:, :7], future_changed[:, :7]),
        'batch_independence_max_abs': maximum_difference(first[:1], alone),
        'batch_independence_matches_course_tolerance': bool(torch.allclose(
            first[:1], alone, atol=1e-5, rtol=1e-5)),
        'state_reset_max_abs': maximum_difference(first, repeated),
        'normalization_max_abs': float(first.logsumexp(-1).abs().max()),
        'training_interface_shape': list(training_logits.shape),
        'training_loss_finite': bool(torch.isfinite(training_loss)),
        'training_gradients_finite_nonzero': bool(gradients and
            all(torch.isfinite(g).all() for g in gradients) and
            sum(g.abs().sum().item() for g in gradients)>0),
    }
    result['passed'] = (
        result['finite']
        and result['causality_max_abs'] <= 1e-6
        and result['batch_independence_matches_course_tolerance']
        and result['state_reset_max_abs'] <= 1e-6
        and result['normalization_max_abs'] <= 1e-3
        and result['training_interface_shape'] == [2, 12, checkpoint['config']['vocab']]
        and result['training_loss_finite']
        and result['training_gradients_finite_nonzero']
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
