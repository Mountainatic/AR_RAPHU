#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gpu_common import atomic_json, sha256_file

CODE_ITEMS = [
    'README_GPU.md', 'RUN_GPU.sh', 'RESUME_GPU.sh', 'SETUP_GPU_ENV.sh',
    'requirements_gpu.txt', 'configs', 'src', 'scripts', 'tests'
]
FORBIDDEN_PARTS = {'.git', '__pycache__', '.pytest_cache', 'shared', 'wandb', 'cache'}
FORBIDDEN_SUFFIXES = {'.xlsx', '.xls', '.tmp', '.pyc'}


def allowed(path: Path, *, keep_best_checkpoints_only: bool) -> bool:
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        return False
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False
    if path.suffix == '.pt':
        if not keep_best_checkpoints_only:
            return False
        return path.name == 'best_model.pt' and 'finalists' in path.parts
    return True


def copy_tree_filtered(source: Path, destination: Path, *, keep_best_checkpoints_only: bool) -> None:
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if not allowed(relative, keep_best_checkpoints_only=keep_best_checkpoints_only):
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def build(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_root).resolve()
    results_root = Path(args.results).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    code_root = output_dir / 'code'
    code_root.mkdir()
    for item in CODE_ITEMS:
        source = source_root / item
        if not source.exists():
            continue
        destination = code_root / item
        if source.is_dir():
            copy_tree_filtered(source, destination, keep_best_checkpoints_only=args.keep_best_checkpoints_only)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    if results_root.is_dir():
        copy_tree_filtered(
            results_root,
            output_dir / 'results_gpu',
            keep_best_checkpoints_only=args.keep_best_checkpoints_only,
        )
    manifest = []
    for path in sorted(output_dir.rglob('*')):
        if path.is_file() and path.name != 'MANIFEST.json':
            manifest.append({
                'path': path.relative_to(output_dir).as_posix(),
                'size': path.stat().st_size,
                'sha256': sha256_file(path),
            })
    atomic_json(output_dir / 'MANIFEST.json', {
        'schema': 'PHYSICS_FIRST_GPU_RESULTS_BUNDLE_V1',
        'file_count': len(manifest),
        'files': manifest,
        'raw_data_included': False,
        'shared_dataset_included': False,
    })
    zip_path = Path(args.zip_path).resolve() if args.zip_path else output_dir.parent / 'PHYSICS_FIRST_GPU_RESULTS_bundle.zip'
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(output_dir.rglob('*')):
            if path.is_file():
                bundle.write(path, output_dir.name / path.relative_to(output_dir))
    digest = sha256_file(zip_path)
    sha_path = zip_path.with_suffix(zip_path.suffix + '.sha256')
    sha_path.write_text(f'{digest}  {zip_path.name}\n', encoding='utf-8')
    return {
        'FINAL_GPU_ZIP': str(zip_path),
        'FINAL_GPU_SHA256': digest,
        'ZIP_SIZE': zip_path.stat().st_size,
        'MANIFEST_FILE_COUNT': len(manifest),
        'PACKAGE_DIR': str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', default=str(ROOT))
    parser.add_argument('--results', default=str(ROOT / 'results_gpu'))
    parser.add_argument('--output-dir', default=str(ROOT / 'return' / 'PHYSICS_FIRST_GPU_RESULTS'))
    parser.add_argument('--zip-path', default=None)
    parser.add_argument('--keep-best-checkpoints-only', action='store_true')
    args = parser.parse_args()
    result = build(args)
    print('GPU_BUNDLE=' + json.dumps(result, ensure_ascii=False), flush=True)
    for key, value in result.items():
        print(f'{key}={value}', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
