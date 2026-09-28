"""Create the clean final checkpoint bundle and verify its file manifest."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = {'.git', 'MP1_final_checkpoint_bundle.zip', 'PACKAGE_MANIFEST.json'}


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    files = sorted(
        path for path in ROOT.rglob('*')
        if path.is_file() and not any(part in EXCLUDED for part in path.relative_to(ROOT).parts)
    )
    records = {
        path.relative_to(ROOT).as_posix(): {
            'bytes': path.stat().st_size,
            'sha256': digest(path),
        }
        for path in files
    }
    frozen = json.loads((ROOT / 'FROZEN_SELECTION.json').read_text(encoding='utf-8'))
    if records['checkpoint/checkpoint.pt']['sha256'] != frozen['checkpoint_sha256']:
        raise ValueError('Checkpoint does not match its pre-test freeze record.')
    manifest = {
        'package': 'MP1_final_train_topk3_clean',
        'protocol': '7506-mp1-wt2-v2',
        'test_status': 'scored_once_after_freeze_for_this_version',
        'files': records,
    }
    manifest_path = ROOT / 'PACKAGE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    bundle_path = ROOT / 'MP1_final_checkpoint_bundle.zip'
    with ZipFile(bundle_path, 'w', compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
        bundle.write(manifest_path, manifest_path.name)
    print(json.dumps({'files': len(files), 'bundle': str(bundle_path),
                      'bundle_bytes': bundle_path.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
