"""Runtime binding rejects stale source, registration, protocol and data."""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.runtime_seal import verify_release,verify_c1
from canonical.cz_h1_posthoc_v22.invariants import ProtocolError,file_sha,source_sha,dump


@pytest.fixture
def release(tmp_path):
    module=tmp_path/'module';module.mkdir()
    authority=tmp_path/'authority';authority.mkdir()
    (module/'engine.py').write_text('frozen = True\n')
    (authority/'solver.py').write_text('numeric_operator = "unchanged"\n')
    (module/'PROTOCOL.json').write_text('{"registered":true}\n')
    sources={'engine.py':source_sha(module/'engine.py')}
    protocols={'PROTOCOL.json':file_sha(module/'PROTOCOL.json')}
    authority_bindings={'solver.py':source_sha(authority/'solver.py')}
    reg=dict(canonical_source_sha256_lf=sources,canonical_protocol_sha256=protocols,
        authority_bindings_sha256_lf=authority_bindings,
        scientific_provenance=dict(revision_status='POST_HOC_PROTOCOL_REVISION',historical_target_test_used_for_protocol_design=True))
    dump(tmp_path/'REGISTRATION.json',reg)
    seal=dict(schema='CZ_CANONICAL_POSTHOC_RELEASE_SEAL_V1',registration_sha256=file_sha(tmp_path/'REGISTRATION.json'),
        canonical_source_sha256_lf=sources,canonical_protocol_sha256=protocols,
        authority_bindings_sha256_lf=authority_bindings)
    dump(tmp_path/'RELEASE_SEAL.json',seal)
    args=(module,authority,tmp_path/'REGISTRATION.json',tmp_path/'RELEASE_SEAL.json',file_sha(tmp_path/'RELEASE_SEAL.json'))
    return args


def test_runtime_seal_verifies_all_bindings_and_posthoc_status(release):
    registration,audit=verify_release(*release)
    assert registration['scientific_provenance']['revision_status']=='POST_HOC_PROTOCOL_REVISION'
    assert audit['canonical_sources_verified']==audit['authority_bindings_verified']==1


@pytest.mark.parametrize('kind',['canonical','authority','registration','protocol','seal'])
def test_runtime_seal_refuses_any_stale_binding(release,kind):
    module,authority,reg,seal,pin=release
    path=dict(canonical=module/'engine.py',authority=authority/'solver.py',registration=reg,
        protocol=module/'PROTOCOL.json',seal=seal)[kind]
    path.write_bytes(path.read_bytes()+b'\nchanged')
    with pytest.raises(ProtocolError):verify_release(*release)


def test_runtime_seal_refuses_an_extra_unregistered_source(release):
    module=release[0]
    (module/'unregistered_import.py').write_text('unsealed = True\n')
    with pytest.raises(ProtocolError,match='inventory changed'):verify_release(*release)


def test_c1_seal_checks_manifest_metadata_base_and_pca(tmp_path):
    sample=Path('sample_ids/train.parquet')
    files=[sample,Path('JOINT_LIFT_PCA_CONTRACT.json'),Path('base_data/cz_czochralski/train.parquet')]
    for index,path in enumerate(files):
        full=tmp_path/path;full.parent.mkdir(parents=True,exist_ok=True);full.write_bytes(str(index).encode())
    manifest=dict(contract_id='frozen_anchor',train_support_hash='IDs',
        train_metadata_parquet_sha256=file_sha(tmp_path/sample),
        pca_sha256=file_sha(tmp_path/files[1]),source_base_sha256=file_sha(tmp_path/files[2]))
    dump(tmp_path/'C1_MAXIMAL_ANCHOR_MANIFEST.json',manifest)
    binding=dict(manifest_sha256=file_sha(tmp_path/'C1_MAXIMAL_ANCHOR_MANIFEST.json'),
        anchor_contract='frozen_anchor',train_support_hash='IDs',sample_relative=sample.as_posix())
    assert verify_c1(tmp_path,binding)==manifest
    (tmp_path/files[1]).write_bytes(b'new PCA')
    with pytest.raises(ProtocolError,match='PCA changed'):verify_c1(tmp_path,binding)
