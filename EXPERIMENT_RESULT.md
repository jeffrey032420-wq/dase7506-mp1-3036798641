# Final Experiment Result

- Predictor: width-192 causal GPT plus static training-derived continuation memory and within-window copying (`student_train_topk3`)
- Frozen checkpoint: `checkpoint/checkpoint.pt`
- SHA-256: `9dc7e4bb6d6bf4690bad10c6087b0406e562ee46f5aadec65b3ea7003993e49b`
- Selected backbone: 25,000 updates × 8 examples × 256 targets = 51,200,000 sampled training targets; seed 31
- Validation BPB: **1.5479098185**, CPU FP32
- Same-weight top-four mechanism ablation: **1.5623286467** validation BPB, CPU FP32
- Frozen full-test BPB: **1.5685106195**, CPU FP32
- Test timing: one full-test evaluation for this version after `FROZEN_SELECTION.json` recorded its checkpoint and implementation hashes; no subsequent model tuning
- Checkpoint: 63,313,762 bytes = 60.381 MiB, below 64 MiB
- CPU time: 40.83 seconds on validation, 3.75× paired local baseline, below 5×
- Peak Windows working set: 1.903 GiB, below 4 GiB
- Exact checkpoint contract: passed; starter standard-library tests: 5/5 passed

The final test score improves on the historical A14 full-test result of 1.6902207397 by 0.1217101201 BPB. The test result was not used to select the final candidate. Full method, training-cost, ablation, and limitation details are in `REPORT.md`.
