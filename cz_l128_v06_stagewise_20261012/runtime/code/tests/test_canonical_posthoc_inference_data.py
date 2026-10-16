import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.data import load_inference,SAMPLE_RELATIVE
from canonical.cz_h1_posthoc_v22.invariants import file_sha,dump,ProtocolError


@pytest.fixture
def synthetic_case(tmp_path):
    pytest.importorskip('pyarrow')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    base=tmp_path/'base_data/cz_czochralski';base.mkdir(parents=True)
    for split,start,end in [('train',0,25),('validation',25,46),('test',46,61)]:
        t=np.arange(start,end)
        pd.DataFrame(dict(entity_id='synthetic',row_in_entity=t,crystal_diameter=t.astype(float)**2)).to_parquet(base/(split+'.parquet'),index=False)
    metadata=tmp_path/SAMPLE_RELATIVE;metadata.mkdir(parents=True)
    t=np.arange(30,34)
    pd.DataFrame(dict(entity_id='synthetic',origin=t,split='validation',
        base_origin_id=['b'+str(i) for i in t],view_sample_id=['v'+str(i) for i in t],
        y_true=-999.)).to_parquet(metadata/'validation.parquet',index=False)
    (metadata/'train.parquet').write_bytes(b'synthetic-train-metadata-only')
    pca=tmp_path/'JOINT_LIFT_PCA_CONTRACT.json';pca.write_text('{}')
    manifest=dict(source_shared=str(tmp_path),contract_id='synthetic',train_support_hash='synthetic',
        train_metadata_parquet_sha256=file_sha(metadata/'train.parquet'),source_base_sha256=file_sha(base/'train.parquet'),
        pca_sha256=file_sha(pca))
    mp=tmp_path/'C1_MAXIMAL_ANCHOR_MANIFEST.json';dump(mp,manifest)
    binding=dict(manifest_sha256=file_sha(mp),anchor_contract='synthetic',train_support_hash='synthetic',
        sample_relative=SAMPLE_RELATIVE+'/train.parquet',inference_source_root=str(tmp_path.resolve()),
        input_bytes_sha256={SAMPLE_RELATIVE+'/validation.parquet':file_sha(metadata/'validation.parquet'),
            'base_data/cz_czochralski/test.parquet':file_sha(base/'test.parquet')})
    context=dict(mode='inference_only',deployment_checkpoint_verified=True,inference_operation='source_validation',
        C1_manifest=manifest,shared=tmp_path,case_id='synthetic',registration={'C1_case_bindings':{'synthetic':binding}},
        protocol={'execution_authorization':dict(source_validation_reporting=True,target_test_inference=False)})
    return context,metadata


def test_source_inference_reconstructs_explicit_h1_and_only_mature_observations(synthetic_case):
    context,_=synthetic_case
    ev,observations,shared,split=load_inference(context,'source_validation')
    assert split=='validation' and np.array_equal(ev.y_true,2*ev.origin-1)
    assert observations.empty
    assert set(ev.latest_available_target_index)=={29,30,31,32}
    assert 'anchor_history_steps' in observations and 'sample_support_contract' in observations


def test_validation_reader_never_projects_target_test_data(synthetic_case,monkeypatch):
    import pyarrow.parquet as pq
    context,_=synthetic_case
    actual=pq.read_table;seen=[]
    def guarded(path,*args,**kwargs):
        seen.append((str(path),kwargs.get('columns')))
        assert Path(path).name!='test.parquet'
        return actual(path,*args,**kwargs)
    monkeypatch.setattr(pq,'read_table',guarded)
    load_inference(context,'source_validation')
    assert all('y_true' not in cols for _,cols in seen)


def test_earliest_inference_row_keeps_empty_history_as_na_not_abort(synthetic_case):
    context,metadata=synthetic_case
    pd.DataFrame(dict(entity_id=['synthetic'],origin=[1],split=['validation'],
        base_origin_id=['b1'],view_sample_id=['v1'])).to_parquet(metadata/'validation.parquet',index=False)
    context['registration']['C1_case_bindings']['synthetic']['input_bytes_sha256'][SAMPLE_RELATIVE+'/validation.parquet']=file_sha(metadata/'validation.parquet')
    ev,observations,_,_=load_inference(context,'source_validation')
    assert len(ev)==1 and ev.y_true.iloc[0]==1. and observations.empty


def test_test_input_byte_drift_rejected_before_label_projection(synthetic_case,monkeypatch):
    import pyarrow.parquet as pq
    context,_=synthetic_case
    path=context['shared']/'base_data/cz_czochralski/test.parquet'
    path.write_bytes(b'replaced-after-release')
    def forbidden(*a,**kw):raise AssertionError('Label reader reached before byte-seal verification')
    monkeypatch.setattr(pq,'read_table',forbidden)
    with pytest.raises(ProtocolError,match='input bytes changed'):load_inference(context,'source_validation')
