#!/usr/bin/env python
"""Run many small independent PyTorch jobs concurrently on one or more GPUs.

For this project a single model is too small for DDP.  Multiple subprocesses on
one device are the intended utilization strategy.  Each job owns an output
record and log; failures do not stop unrelated jobs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def telemetry_loop(devices, output: Path, stop: threading.Event, interval: float):
    fields = ["timestamp", "gpu", "utilization_gpu_percent", "memory_used_mib",
              "memory_total_mib", "temperature_c", "power_w"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            for gpu in devices:
                try:
                    command = [
                        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits", "-i", str(gpu),
                    ]
                    text = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
                    util, used, total, temp, power = [part.strip() for part in text.split(",")]
                    writer.writerow({
                        "timestamp": time.time(), "gpu": gpu,
                        "utilization_gpu_percent": util, "memory_used_mib": used,
                        "memory_total_mib": total, "temperature_c": temp, "power_w": power,
                    })
                except Exception:
                    writer.writerow({"timestamp": time.time(), "gpu": gpu})
            handle.flush()
            stop.wait(interval)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--devices", default="0")
    p.add_argument("--workers-per-device", type=int, default=3)
    p.add_argument("--workdir", default=".")
    p.add_argument("--records-dir", default="results_stage1/STAGE1_DUAL_SOLVER_V20/job_records")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--telemetry-interval", type=float, default=1.0)
    args = p.parse_args()
    if args.workers_per_device < 1:
        raise ValueError("workers-per-device must be positive")
    root = Path(args.workdir).resolve()
    manifest_path = Path(args.manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    jobs = payload["jobs"]
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    records = root / args.records_dir / manifest_path.stem
    records.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for index, job in enumerate(jobs):
        job_id = str(job["job_id"])
        done = records / job_id / "DONE.json"
        if args.resume and done.exists():
            try:
                prior = json.loads(done.read_text(encoding="utf-8"))
                if prior.get("returncode") == 0:
                    continue
            except Exception:
                pass
        pending.put((index, job))

    stop = threading.Event()
    telemetry = threading.Thread(
        target=telemetry_loop,
        args=(devices, records / "gpu_telemetry.csv", stop, args.telemetry_interval),
        daemon=True,
    )
    telemetry.start()
    result_lock = threading.Lock()
    results = []

    def worker(device: str, slot: int):
        while True:
            try:
                index, job = pending.get_nowait()
            except queue.Empty:
                return
            job_id = str(job["job_id"])
            job_dir = records / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            lock = job_dir / ".lock"
            # A stale lock from a crashed prior orchestrator must not block resume.
            # This process owns the in-memory queue, so no second live worker can
            # receive the same job.
            if lock.exists():
                lock.unlink()
            lock.write_text(json.dumps({"pid": os.getpid(), "device": device,
                                        "slot": slot, "started": time.time()}), encoding="utf-8")
            command = [str(part) for part in job["command"]]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            started = time.time()
            with (job_dir / "stdout.log").open("w", encoding="utf-8") as stdout, \
                 (job_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
                process = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
            record = {
                "job_id": job_id, "index": index, "command": command,
                "physical_gpu": device, "worker_slot": slot,
                "returncode": process.returncode,
                "started": started, "finished": time.time(),
                "runtime_seconds": time.time() - started,
            }
            atomic_json(job_dir / "DONE.json", record)
            lock.unlink(missing_ok=True)
            with result_lock:
                results.append(record)
            pending.task_done()

    threads = []
    for device in devices:
        for slot in range(args.workers_per_device):
            thread = threading.Thread(target=worker, args=(device, slot), daemon=True)
            thread.start(); threads.append(thread)
    for thread in threads:
        thread.join()
    stop.set(); telemetry.join(timeout=3)
    results.sort(key=lambda item: item["index"])
    summary = {
        "manifest": str(manifest_path), "devices": devices,
        "workers_per_device": args.workers_per_device,
        "submitted": len(jobs), "executed": len(results),
        "failed": sum(item["returncode"] != 0 for item in results),
        "results": results,
    }
    atomic_json(records / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    raise SystemExit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
