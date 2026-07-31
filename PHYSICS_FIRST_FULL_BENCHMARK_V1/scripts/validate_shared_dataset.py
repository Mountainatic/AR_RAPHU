#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gpu_common import atomic_json
from src.gpu_data import validate_shared_dataset

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared', required=True)
    parser.add_argument('--output', default=str(ROOT / 'results_gpu' / 'SHARED_VALIDATION.json'))
    args = parser.parse_args()
    report = validate_shared_dataset(args.shared)
    atomic_json(args.output, report)
    print('SHARED_VALIDATION=' + json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'PASS' else 2

if __name__ == '__main__':
    raise SystemExit(main())
