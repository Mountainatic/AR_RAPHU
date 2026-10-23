from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
for name,v in m.items():
    rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
    p=ROOT/rel;assert p.is_file() and p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],name
r=json.loads((ROOT/'review'/'COMBINED_REVIEW.json').read_text(encoding='utf-8'));assert r['final_configurations']==45 and r['candidate_A_searches']==186 and len(r['joint_development_successes'])==2 and not r['source_final_numeric_read']
print('SNAPSHOT_VERIFIED',len(m),'files; 45 configurations; final unopened')
