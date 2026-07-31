#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gpu_runner import aggregate_results

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=str(ROOT / 'results_gpu'))
    args = parser.parse_args()
    summary = aggregate_results(args.results)
    print('GPU_AGGREGATE=' + json.dumps({'rows': len(summary)}, ensure_ascii=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
