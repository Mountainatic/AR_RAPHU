#!/usr/bin/env python3
"""Fail-fast hardware, CUDA, MPS, environment, and privacy preflight."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.runtime_environment import require_runtime_environment  # noqa: E402


def command_output(command: list[str]) -> str:
    return subprocess.check_output(
        command, text=True, stderr=subprocess.STDOUT
    ).strip()


def configure_runtime_threads() -> None:
    intraop = int(os.environ.get("AR_RAPHU_TORCH_THREADS", "1"))
    interop = int(os.environ.get("AR_RAPHU_TORCH_INTEROP_THREADS", "1"))
    if intraop < 1 or interop < 1:
        raise ValueError("PyTorch thread limits must be positive.")
    torch.set_num_threads(intraop)
    torch.set_num_interop_threads(interop)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-mps", action="store_true")
    args = parser.parse_args()
    manager = require_runtime_environment()
    configure_runtime_threads()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    properties = torch.cuda.get_device_properties(device)
    memory_mib = properties.total_memory // (1024 * 1024)
    if "5090" not in name:
        raise RuntimeError(f"Expected RTX 5090, found {name!r}.")
    if memory_mib < 30000:
        raise RuntimeError(f"Expected at least 30000 MiB GPU memory, found {memory_mib}.")
    if properties.major < 12:
        raise RuntimeError(
            f"Expected Blackwell compute capability >= 12.0, found "
            f"{properties.major}.{properties.minor}."
        )
    if args.require_mps:
        pipe = Path(os.environ.get("CUDA_MPS_PIPE_DIRECTORY", ""))
        pid_file = pipe / "nvidia-cuda-mps-control.pid"
        if not pipe.is_dir() or not pid_file.is_file():
            raise RuntimeError("MPS control daemon PID file is missing.")
    torch.manual_seed(20260725)
    left = torch.randn(2048, 2048, device=device)
    right = torch.randn(2048, 512, device=device)
    checksum = float((left @ right).float().mean().item())
    torch.cuda.synchronize(device)
    private_source = PROJECT_ROOT / "实验数据1.xlsx"
    if private_source.exists():
        raise RuntimeError("Private CZ source must not be present in the AutoDL package.")
    payload = {
        "status": "COMPLETED",
        "unix_time": time.time(),
        "runtime_manager": manager,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_compiled_cuda": torch.version.cuda,
        "gpu": name,
        "compute_capability": [properties.major, properties.minor],
        "gpu_memory_mib": memory_mib,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        "mps_log_directory": os.environ.get("CUDA_MPS_LOG_DIRECTORY"),
        "mps_required": args.require_mps,
        "cuda_test_checksum": checksum,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "private_cz_present": False,
    }
    output = PROJECT_ROOT / "results" / "runtime" / "autodl_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
