"""New immutable scientific release; byte-identical numerical reuse receipts."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import json,shutil,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path('/root/autodl-tmp/CZ_CROSSROD_L128_V06_20261010')
CASES=('Rod_1_to_Rod_2','Rod_2_to_Rod_1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n')
def prepare():
    if (ROOT/'release').exists():raise RuntimeError('Immutable release already exists')
    oldpin=json.loads((OLD/'release/RELEASE_PIN.json').read_text())
    assert sha(OLD/'release/RELEASE_SEAL.json')==oldpin['release_sha256']
    # Numeric K sources/configs must be EXACTLY unchanged. New stagewise code
    # lives in new modules; it cannot make an old K fit eligible by relabeling.
    source_bindings={}
    for p in (OLD/'code').rglob('*'):
        if not p.is_file() or p.suffix not in {'.py','.json'}:continue
        rel=p.relative_to(OLD/'code')
        if rel.as_posix()=='canonical/cz_crossrod_l128_20261008/CANONICAL_PROTOCOL.json':continue
        new=ROOT/'code'/rel
        if not new.exists() or sha(p)!=sha(new):raise RuntimeError('Inherited numerical source differs: '+str(rel))
        source_bindings[rel.as_posix()]=sha(p)
    required=json.loads((OLD/'STAGEWISE_REQUIRED_K_REUSE_AUDIT.json').read_text())
    assert all(not v['missing'] and v['required_K_models']==v['available'] for v in required.values())
    receipt=ROOT/'K_REUSE_COMPATIBILITY.json';shutil.copy2(OLD/'STAGEWISE_REQUIRED_K_REUSE_AUDIT.json',receipt)
    write(ROOT/'INHERITED_NUMERICAL_SOURCE_COMPATIBILITY.json',dict(old_release=oldpin['release_sha256'],exact_byte_bindings=source_bindings))
    shutil.copytree(OLD/'C1',ROOT/'C1',copy_function=os.link,ignore=shutil.ignore_patterns('*.json'))
    for p in (OLD/'C1').rglob('*.json'):
        q=ROOT/'C1'/p.relative_to(OLD/'C1');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for case in CASES:
        shared=ROOT/'C1'/case
        assert not (shared/'base_data/cz_czochralski/test.parquet').exists()
        assert not (shared/'TARGET_DATA_SEAL.json').exists()
        f=shared/'C1_MAXIMAL_ANCHOR_MANIFEST.json';m=json.loads(f.read_text());m['source_shared']=str(shared);write(f,m)
        # All data/sample IDs/label construction remain identical.
        for p in (OLD/'C1'/case).rglob('*.parquet'):
            assert sha(p)==sha(ROOT/'C1'/case/p.relative_to(OLD/'C1'/case))
    # Baselines stay sealed under their ORIGINAL registration. A distinct
    # import receipt binds them to this release, rather than rewriting seals.
    baseline=json.loads((OLD/'BASELINE_RELEASE.json').read_text())
    artifacts={str(OLD/'BASELINE_RELEASE.json'):sha(OLD/'BASELINE_RELEASE.json')}
    for p,h in baseline['file_sha256'].items():
        assert sha(p)==h;artifacts[p]=h
    jobs=json.loads((ROOT/'prepared/BASELINE_JOBS.json').read_text())
    assert jobs==json.loads((OLD/'prepared/BASELINE_JOBS.json').read_text())
    manifests=[OLD/'baselines'/c/'cpu_verified/CPU_FREEZE_MANIFEST.json' for c in CASES]+[OLD/'baselines/neural/NEURAL_FREEZE_MANIFEST.json']
    records=[]
    for path in manifests:
        artifacts[str(path)]=sha(path);records.extend(json.loads(path.read_text())['records'])
    assert len(records)==len(jobs)==46
    identity=lambda r:(r['direction'],r['information_set'],r['model'])
    assert {identity(r) for r in records}=={identity(j) for j in jobs}
    cpu=neural=seeds=na=0
    for r in records:
        assert r['history_steps']==128 and r['h_steps']==1 and r['test_accessed'] is False
        if r['status']=='FROZEN':
            if r['family']=='CPU_BASELINE':
                cpu+=1
                for p in Path(r['frozen']['checkpoint_dir']).rglob('*'):
                    if p.is_file():artifacts[str(p)]=sha(p)
                assert sha(r['validation_path'])==r['validation_sha256'];artifacts[r['validation_path']]=r['validation_sha256']
            else:
                neural+=1;assert sorted(s['seed'] for s in r['seeds'])==list(range(20260817,20260827))
                artifacts[r['selection_path']]=sha(r['selection_path'])
                for s in r['seeds']:
                    assert sha(s['checkpoint'])==s['sha256'];artifacts[s['checkpoint']]=s['sha256'];seeds+=1
                    assert sha(s['validation'])==s['validation_sha256'];artifacts[s['validation']]=s['validation_sha256']
        else:
            assert r['status']=='NOT_RUN_PROTOCOL_INCOMPATIBLE' and r['model'] in {'SEASONAL_PERSISTENCE','N4SID'};na+=1
    assert (cpu,neural,seeds,na)==(28,12,120,6)
    shutil.copy2(OLD/'BASELINE_RELEASE.json',ROOT/'BASELINE_RELEASE.json')
    os.symlink(OLD/'baselines',ROOT/'baselines',target_is_directory=True)
    module=ROOT/'code/canonical/cz_crossrod_l128_20261008';f=module/'CANONICAL_PROTOCOL.json';protocol=json.loads(f.read_text())
    protocol.update(revision_id='CZ_CROSSROD_L128_STAGEWISE_V06_20261012',revision='CZ_CROSSROD_L128_STAGEWISE_V06_20261012',
        user_confirmation='2026-10-12: 在 GitHub 逐层实现基础上修改、重跑并复用匹配的既有训练')
    protocol['final_development_cf']=dict(coverage='FOUR_FINITE_PARAMETER_CF_BLOCKS_WITH_CONTEXT',support_rule='SAVED_FINITE_STAGEWISE_CF',all_source_anchors_required=False,recursive_prefix_replay=False)
    protocol['stagewise_migration']=dict(schema='CZ_FINITE_STAGEWISE_PROTOCOL_V1',github_branch='prism-unified-hw-r2-protocols-20260921',github_commit='6ba6f1b510cc3476f9dbd1c64e86b07710b239da',
        legacy_root=str(OLD),legacy_policy_id=oldpin['policy_id'],legacy_release_sha256=oldpin['release_sha256'],
        K_reuse_receipt=str(receipt),K_reuse_receipt_sha256=sha(receipt),
        external_outer='ALL_RECIPE_SELECTION_AND_PARAMETER_FITS_INSIDE_OUTER_FIT_PARTITION',
        internal_upstream='FIXED_RECIPES_FOLD_LOCAL_PARAMETERS_AND_PCA',fully_nested_hyperparameter_CF=False,
        full_K_registry_unchanged=True,C_W_A_old_models_imported=False,
        scientific_equivalence_to_old_recursive_protocol=False,
        interpretation='OUTER_HOLDOUT_STAGEWISE_POLICY_EVALUATION; INNER_UPSTREAM_RECIPE_CF_NOT_FULLY_NESTED')
    write(f,protocol)
    sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/src')]
    from canonical.cz_crossrod_l128_20261008.release import build_release
    pin=build_release(ROOT/'code',ROOT/'C1',ROOT/'release')
    write(ROOT/'BASELINE_IMPORT_RECEIPT.json',dict(status='EXACT_BASELINE_REUSE_VERIFIED',new_main_release_sha256=pin,
        source_root=str(OLD),source_baseline_release_sha256=sha(OLD/'BASELINE_RELEASE.json'),
        cpu_frozen=cpu,neural_tasks=neural,neural_seeds=seeds,protocol_NA=na,
        same_L128_H1_data_sample_ids_candidates_seeds_numerical_source=True,artifact_sha256=artifacts,target_test_accessed=False))
    write(ROOT/'MIGRATION_PREPARATION.json',dict(status='SEALED_ACCEPTANCE_PENDING',new_release_sha256=pin,
        K_reused_per_direction={c:required[c]['available'] for c in CASES},CPU_tasks_reused=cpu,neural_seed_checkpoints_reused=seeds,
        C_W_A_recomputed=True,legacy_paused=True,target_test_accessed=False))
    print(pin,flush=True)
if __name__=='__main__':prepare()
