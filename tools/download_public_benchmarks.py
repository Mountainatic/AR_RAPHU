#!/usr/bin/env python3
"""Download PB1 sources outside the repository and write immutable lineage."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import nonlinear_benchmarks as nb

from ar_raphu.datasets.lineage import build_lineage, write_lineage


WHPN_URL = (
    "https://data.4tu.nl/file/"
    "1f194001-affa-4459-870a-ad9e9d9146f9/"
    "2dbbc046-1ac2-43b2-bf4e-53b5a4be8b96"
)


def _download_via_official_loader(dataset_id: str, raw_root: Path) -> str:
    kwargs = {
        "data_file_locations": True,
        "dir_placement": str(raw_root),
        "force_download": False,
    }
    if dataset_id == "pwh":
        locations = (Path(nb.ParWH(**kwargs)),)
        archive = raw_root / "ParWH" / "ParWHFiles.zip"
    elif dataset_id == "cascaded_tanks":
        locations = (Path(nb.Cascaded_Tanks(**kwargs)),)
        archive = raw_root / "Cascaded_Tanks" / "CascadedTanksFiles.zip"
    elif dataset_id == "silverbox":
        locations = tuple(Path(p) for p in nb.Silverbox(**kwargs))
        archive = raw_root / "Silverbox" / "SilverboxFiles.zip"
    else:
        raise ValueError(dataset_id)
    if any(not path.is_file() for path in locations):
        if not archive.is_file():
            raise FileNotFoundError(
                f"Official loader returned absent files and archive is missing: {archive}"
            )
        with zipfile.ZipFile(archive) as handle:
            handle.testzip()
            handle.extractall(archive.parent)
    absent = [str(path) for path in locations if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Expected extracted files are absent: {absent}")
    if len(locations) == 1:
        return str(locations[0])
    return json.dumps([str(path) for path in locations])


def _download_whpn(raw_root: Path) -> str:
    target_dir = raw_root / "WHPN"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "WienerHammersteinFiles.zip"
    if not target.exists():
        temporary = target.with_suffix(".zip.part")
        urllib.request.urlretrieve(WHPN_URL, temporary)
        temporary.replace(target)
    return str(target)


SOURCE = {
    "pwh": {
        "url": "https://data.4tu.nl/articles/dataset/12950081/1",
        "doi": "10.4121/12950081.v1",
        "version": "1",
        "license": "CC BY-SA 4.0",
        "loader": "nonlinear_benchmarks.ParWH",
    },
    "whpn": {
        "url": "https://data.4tu.nl/datasets/1f194001-affa-4459-870a-ad9e9d9146f9/2",
        "doi": "10.4121/12952124.v2",
        "version": "2",
        "license": "CC BY-SA 4.0",
        "loader": "project_archive_audit",
    },
    "cascaded_tanks": {
        "url": "https://data.4tu.nl/articles/dataset/12960104/1",
        "doi": "10.4121/12960104.v1",
        "version": "1",
        "license": "CC BY-SA 4.0",
        "loader": "nonlinear_benchmarks.Cascaded_Tanks",
    },
    "silverbox": {
        "url": "https://www.nonlinearbenchmark.org/benchmarks/silverbox",
        "doi": "10.23919/ECC.2013.6669201",
        "version": "official_2026-07-27",
        "license": "NOT_STATED_ON_SOURCE",
        "loader": "nonlinear_benchmarks.Silverbox",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/root/OPS_UOI_WORKSPACE/data/raw"),
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data_manifests/public_benchmarks"),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(SOURCE),
        required=True,
    )
    args = parser.parse_args()
    args.raw_root.mkdir(parents=True, exist_ok=True)
    for dataset_id in args.dataset:
        before = set(args.raw_root.rglob("*"))
        location = (
            _download_whpn(args.raw_root)
            if dataset_id == "whpn"
            else _download_via_official_loader(dataset_id, args.raw_root)
        )
        dataset_dir = {
            "pwh": args.raw_root / "ParWH",
            "whpn": args.raw_root / "WHPN",
            "cascaded_tanks": args.raw_root / "Cascaded_Tanks",
            "silverbox": args.raw_root / "Silverbox",
        }[dataset_id]
        files = [p for p in dataset_dir.rglob("*") if p.is_file()]
        if not files:
            raise RuntimeError(f"No files found after downloading {dataset_id}.")
        source = SOURCE[dataset_id]
        payload = build_lineage(
            dataset_id=dataset_id,
            raw_root=args.raw_root,
            files=files,
            official_source=source["url"],
            doi=source["doi"],
            version_id=source["version"],
            license_name=source["license"],
            loader=source["loader"],
        )
        payload["download_return"] = location
        payload["new_file_count"] = len(set(args.raw_root.rglob("*")) - before)
        write_lineage(args.manifest_root / f"{dataset_id}.json", payload)
        print(f"{dataset_id}: {len(files)} files -> {location}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
