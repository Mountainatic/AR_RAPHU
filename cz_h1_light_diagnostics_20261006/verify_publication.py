"""Standard-library check of this public archive; no private inputs or training."""
from pathlib import Path
import ast
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent
FORBIDDEN = {'.xlsx', '.xls', '.xlsm', '.xlsb', '.parquet', '.arrow', '.feather',
             '.npz', '.npy', '.pt', '.pth', '.ckpt', '.pkl', '.pickle', '.h5',
             '.hdf5', '.zip', '.7z', '.tar', '.gz', '.bundle', '.pyc'}
ROWWISE_FIELDS = {'y_true', 'y_target', 'y_base_pred', 'y_final_pred',
                  'r_base', 'r_final', 'base_residual', 'final_residual',
                  'base_prediction', 'final_prediction', 'crystal_diameter'}

def check(root=ROOT):
    manifest = json.loads((root / 'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
    expected = {item['path'] for item in manifest['files']}
    expected.add('PUBLICATION_MANIFEST.json')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    assert actual == expected, f'Unexpected/missing files: {actual ^ expected}'
    csv_count = 0
    for item in manifest['files']:
        relative = Path(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root / relative
        assert not path.is_symlink()
        assert path.suffix.lower() not in FORBIDDEN, str(relative)
        data = path.read_bytes()
        assert len(data) == item['bytes'], str(relative)
        assert hashlib.sha256(data).hexdigest() == item['sha256'], str(relative)
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=str(relative))
        elif path.suffix == '.json':
            json.loads(data.decode('utf-8-sig'))
        elif path.suffix == '.png':
            assert data[:8] == b'\x89PNG\r\n\x1a\n', str(relative)
        elif path.suffix == '.csv':
            rows = list(csv.reader(data.decode('utf-8-sig').splitlines()))
            assert rows and not (set(rows[0]) & ROWWISE_FIELDS), str(relative)
            assert all(len(row) == len(rows[0]) for row in rows[1:]), str(relative)
            if 'target' in rows[0]:
                assert path.name in {'COHERENCE_SPECTRA.csv', 'CONTROL_OUTPUT_COHERENCE.csv', 'ROTATION_PHASE_HELDOUT.csv'}
                target_index = rows[0].index('target')
                assert all(row[target_index] in {'dD', 'base_residual'} for row in rows[1:]), str(relative)
            csv_count += 1
    print(f"PASS: {len(manifest['files']) + 1} public files, {csv_count} aggregate CSVs; hashes/schema/source syntax checked; no private inputs accessed.")

if __name__ == '__main__':
    check()
