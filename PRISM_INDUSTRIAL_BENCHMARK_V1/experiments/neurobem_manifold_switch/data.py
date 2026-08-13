from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from prism_benchmark.neurobem_literature import LiteratureTrajectory, read_neurobem_csv, resample_track_b_100hz


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class NeuroBEMSource:
    """Resolve the frozen release identities without copying raw data."""

    def __init__(self, data_root: Path, split_manifest: Path, release_root: Path | None = None):
        self.data_root = Path(data_root)
        self.manifest_path = Path(split_manifest)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.processed = self.data_root / "extracted" / "processed_data"
        self.release_root = None if release_root is None else Path(release_root)
        if not self.processed.is_dir():
            raise FileNotFoundError(f"NEUROBEM_PROCESSED_DATA_NOT_FOUND:{self.processed}")

    def names(self, partition: str) -> list[str]:
        key = {"train": "train_trajectories", "validation": "validation_trajectories", "test": "test_trajectories"}[partition]
        return list(self.manifest[key])

    def path(self, name: str) -> Path:
        candidate = self.processed / name
        if not candidate.is_file() and self.release_root is not None:
            matches = list(self.release_root.glob(f"*/{name}"))
            if len(matches) == 1:
                candidate = matches[0]
        if not candidate.is_file():
            raise FileNotFoundError(f"FROZEN_TRAJECTORY_MISSING:{name}")
        return candidate

    def load(self, partition: str, names: list[str] | None = None) -> list[LiteratureTrajectory]:
        selected = self.names(partition) if names is None else names
        return [LiteratureTrajectory(name, partition, resample_track_b_100hz(read_neurobem_csv(self.path(name)))) for name in selected]

    def audit(self) -> dict[str, object]:
        counts = {part: len(self.names(part)) for part in ("train", "validation", "test")}
        missing = {part: [n for n in self.names(part) if not self._exists(n)] for part in counts}
        return {
            "split_manifest": str(self.manifest_path),
            "split_manifest_sha256": _sha(self.manifest_path),
            "processed_root": str(self.processed),
            "release_counts": counts,
            "paper_counts": self.manifest.get("paper_counts"),
            "release_count_discrepancy_retained": not bool(self.manifest.get("release_counts_match_paper_counts")),
            "missing": missing,
            "all_frozen_identities_present": not any(missing.values()),
            "entity_isolation": "ONE_CSV_ONE_TRAJECTORY_NO_CROSS_BOUNDARY_HISTORY",
        }

    def train_parent_holdout(self, fit_fraction: float = 0.75) -> tuple[list[str], list[str], dict[str, object]]:
        names = self.names("train")
        parents = sorted({name.rsplit("_seg_", 1)[0] for name in names})
        cut = int(np.ceil(len(parents) * fit_fraction))
        fit_parents, calibration_parents = set(parents[:cut]), set(parents[cut:])
        test_hashes = {self._file_sha(self.path(name)): name for name in self.names("test")}
        collisions = []
        clean = []
        for name in names:
            digest = self._file_sha(self.path(name))
            if digest in test_hashes:
                collisions.append({"train": name, "test": test_hashes[digest], "sha256": digest})
            else:
                clean.append(name)
        fit = [name for name in clean if name.rsplit("_seg_", 1)[0] in fit_parents]
        calibration = [name for name in clean if name.rsplit("_seg_", 1)[0] in calibration_parents]
        return fit, calibration, {
            "contract": "CHRONOLOGICAL_PARENT_FLIGHT_TRAIN_FIT_CALIBRATION_R2",
            "fit_fraction": fit_fraction, "parent_count": len(parents), "fit_parent_count": len(fit_parents),
            "calibration_parent_count": len(calibration_parents), "fit_segments": len(fit),
            "calibration_segments": len(calibration), "excluded_test_sha_collisions": collisions,
            "test_sha_excluded_from_fit_and_calibration": True,
        }

    @staticmethod
    def _file_sha(path: Path) -> str:
        return _sha(path)

    def _exists(self, name: str) -> bool:
        if (self.processed / name).is_file():
            return True
        return self.release_root is not None and any(self.release_root.glob(f"*/{name}"))
