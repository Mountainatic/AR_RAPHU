from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.gpu_common import atomic_json, atomic_npz


def test_atomic_writers_use_unique_temporary_paths(tmp_path):
    json_path = tmp_path / "shared.json"
    npz_path = tmp_path / "shared.npz"

    def write(index: int) -> None:
        atomic_json(json_path, {"index": index})
        atomic_npz(npz_path, index=np.asarray([index], dtype=np.int64))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write, range(200)))

    assert json_path.is_file()
    with np.load(npz_path) as archive:
        assert archive["index"].shape == (1,)
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))
