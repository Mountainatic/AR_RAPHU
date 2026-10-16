"""Target values are reached only through the all-model transfer freeze."""
import hashlib
import json
from pathlib import Path


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_target_data(shared):
    shared=Path(shared)
    seal=json.loads((shared/'TARGET_DATA_SEAL.json').read_text())
    global_path=Path(seal['global_freeze_path'])
    if file_sha(global_path)!=seal['global_freeze_sha256']:raise RuntimeError('Global transfer freeze changed')
    freeze=json.loads(global_path.read_text())
    if freeze['status']!='ALL_MAIN_AND_BASELINES_FROZEN' or freeze['test_used_for_selection'] is not False:
        raise RuntimeError('Both-direction model freeze is incomplete')
    for path,expected in freeze['artifact_sha256'].items():
        if file_sha(path)!=expected:raise RuntimeError('Frozen model/checkpoint changed')
    if file_sha(shared/'base_data/cz_czochralski/test.parquet')!=seal['target_base_sha256']:
        raise RuntimeError('Frozen target data changed')
    return seal
