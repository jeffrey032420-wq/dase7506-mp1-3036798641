# MP1 Final Submission Checklist

## Final score to submit

- Student ID: fill in on the course website
- Full-test BPB: **1.5685106195**
- Precision/device: CPU FP32
- Checkpoint SHA-256: `9dc7e4bb6d6bf4690bad10c6087b0406e562ee46f5aadec65b3ea7003993e49b`

## Upload

1. Push the submission code, data, report, result JSON files, and manifest to an immutable code repository. Do not upload exploratory `code/runs/` or discarded checkpoint directories; the final bundle already contains the exact submitted checkpoint and runnable code.
2. Upload `MP1_final_checkpoint_bundle.zip` as the matching checkpoint bundle.
3. On the course website, submit the student ID, the full-test BPB above, the
   code-repository link, and the checkpoint-bundle link.
4. Keep the repository and checkpoint bytes unchanged after submission.

## Included evidence

- `REPORT.md`: method, controlled comparisons, ablation, limitations, resources
- `FROZEN_METRICS.json`: final score and exact checkpoint identity
- `FROZEN_SELECTION.json`: pre-test checkpoint and code identity
- `results/final-test-cpu-fp32.json`: official final test output
- `results/final-contract-check.json`: exact frozen model contract
- `results/final-resource-measurement.json`: CPU time and peak memory
- `PACKAGE_MANIFEST.json`: hashes for the final submission files

## Final local reproduction

```bash
cd code
python -m unittest discover -s tests -v
python verify_checkpoint.py --checkpoint ../checkpoint/checkpoint.pt --threads 4
python evaluate.py --checkpoint ../checkpoint/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```
