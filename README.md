# MP1 final submission — 1.5685106195 BPB

This repository contains the final frozen submission for DASE7506 MP1 (Student
ID: **3036798641**). The
reported full-test score is **1.5685106195 BPB**, evaluated using the supplied
WikiText-2 scorer on CPU FP32.

## Submission links

- Checkpoint bundle: [MP1_final_checkpoint_bundle.zip](https://github.com/jeffrey032420-wq/MP1-improvement/releases/download/mp1-final-1.5685106195/MP1_final_checkpoint_bundle.zip)
- Release record: [mp1-final-1.5685106195](https://github.com/jeffrey032420-wq/MP1-improvement/releases/tag/mp1-final-1.5685106195)
- Exact checkpoint SHA-256: `9dc7e4bb6d6bf4690bad10c6087b0406e562ee46f5aadec65b3ea7003993e49b`

## Reproduce the frozen score

Install the dependencies, then run the following from `code/`:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_checkpoint.py --checkpoint ../checkpoint/checkpoint.pt --threads 4
python evaluate.py --checkpoint ../checkpoint/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

The final checkpoint, evaluation result, resource measurement, pre-test freeze
record, package manifest, and report are stored at the repository root.
`code/README.md` explains the final model and training record in detail.

## Compliance summary

The final predictor learns only from the supplied training text. The supplied
tokenizer, data, and evaluator are unchanged. Evaluation uses independent
causal 256-token windows without cross-window state. The checkpoint is 60.381
MiB, measured peak working set is 1.903 GiB, and measured CPU scoring time is
3.75 times the paired baseline; all are within the assignment limits.

## AI assistance disclosure

OpenAI Codex substantially assisted with implementation, validation-only
experiments, reproducibility checks, and documentation. The student remains
responsible for understanding, verifying, and explaining the work.
