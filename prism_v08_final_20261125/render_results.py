"""Rebuild tables and figures from the released aggregates."""

import importlib.util
from pathlib import Path
import shutil
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python render_results.py NEW_DIRECTORY")
    release = Path(__file__).resolve().parent
    renderer_path = release / "workflow/PRISM_V08_FINAL_EVIDENCE_20261125/make_artifacts.py"
    spec = importlib.util.spec_from_file_location("prism_evidence_renderer", renderer_path)
    renderer = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(renderer)

    target = Path(sys.argv[1]).resolve()
    target.mkdir(parents=True, exist_ok=False)
    for name in ("TAIL_RESULTS.json", "BENCH_PARETO.json", "BENCH_SCALING.json"):
        shutil.copyfile(release / "results" / name, target / name)
    renderer.ROOT = target
    renderer.main()


if __name__ == "__main__":
    main()
