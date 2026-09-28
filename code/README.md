# MP1 final submission — reproducible frozen checkpoint

This repository contains only the final frozen predictor and the material
needed to evaluate it. It uses the supplied WikiText-2 files, BPE-2048
tokenizer, and unchanged scorer. No external data, pretrained weights,
test-based tuning, cross-window state, or evaluation network access is used.

## Result

- Full-test BPB: **1.5685106195** (CPU FP32)
- Validation BPB: 1.5479098185 (CPU FP32)
- Checkpoint SHA-256: `9dc7e4bb6d6bf4690bad10c6087b0406e562ee46f5aadec65b3ea7003993e49b`
- Predictor module: `student_train_topk3`

The selected checkpoint and implementation hashes were recorded before the
full test in `FROZEN_SELECTION.json`; the test score was not used to change
the predictor.

## Installation and exact evaluation

Use Python 3.12 and install CPU PyTorch plus the pinned requirements. From
`code/`:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_checkpoint.py --checkpoint ../checkpoint/checkpoint.pt --threads 4
python evaluate.py --checkpoint ../checkpoint/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

The last command must reproduce the fixed test protocol. Small machine-level
floating-point differences are possible, but the checkpoint, tokenizer, scorer
and code hashes are recorded in the result JSON and `PACKAGE_MANIFEST.json`.
No retraining is needed to reproduce the submitted score.

## Final method and training record

The model is a six-block width-192 causal GPT with RMSNorm, SwiGLU, untied
embeddings, a frozen train-text bigram residual, static train-only n-gram
continuation tables, two- and three-token top-four conditional mixtures, and
strictly within-window causal copying. All static tables are serialized in
`checkpoint/checkpoint.pt`. `student_train_topk3.py` and its imported
`student_*.py` modules are the complete inference dependency chain.

The selected neural backbone was trained from scratch on the supplied training
split with seed 31: 25,000 updates, batch 8, 256 targets per example,
51,200,000 sampled targets, peak learning rate 0.0007, 200-step warmup,
cosine decay to 1% of peak, beta1 0.9, beta2 0.95, weight decay 0.1 and
gradient clipping at 1. The source is `train_large_memory.py`; it consumes
only `train_only_data.py`, which verifies and reads the supplied training text.
The static tables are constructed from the same training split, then combined
with the backbone by `graft_large_backbone.py`; `half_storage.py` converts
stored learned tensors to FP16 while CPU evaluation loads FP32 modules.
`prepare_topk_calibrated.py` applies validation-selected mixture strengths of
0.1 for the two-token and three-token tables. Full experimental lineage,
costs, and limitations are disclosed in `REPORT.md`.

## Constraints and evidence

The checkpoint is 60.381 MiB (under 64 MiB); CPU FP32 validation takes 40.83
seconds, 3.75 times the paired 10.90-second local baseline (under 5 times);
and peak Windows working set is 1.903 GiB (under 4 GiB). The frozen checkpoint
passes finite-output, causality, state-reset, normalization, batch-independence,
and training-interface checks. `results/` contains the test, resource, and
ablation records. `REPORT.md` includes the supplied baseline, equal-budget
architecture comparison, and same-weight mechanism ablation.

## AI assistance disclosure

OpenAI Codex substantially assisted with inspecting the starter package,
planning and implementing model and memory changes, running validation-only
experiments, constraint verification, and documentation. The student is
responsible for understanding, independently verifying, and explaining the
submission.
