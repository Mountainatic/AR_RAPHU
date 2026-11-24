"""Pinned release verification before data loading or model fitting."""
import json
from pathlib import Path
from .invariants import ProtocolError,file_sha,source_sha,digest


def _inside(root,relative):
    root=Path(root).resolve()
    path=(root/relative).resolve()
    if not path.is_relative_to(root):
        raise ProtocolError('Seal path escapes its registered code root')
    return path


def verify_release(module,authority,registration_path,release_seal_path,expected_release_sha):
    module,authority,registration_path,release_seal_path=map(Path,(module,authority,registration_path,release_seal_path))
    if not expected_release_sha or file_sha(release_seal_path)!=expected_release_sha:
        raise ProtocolError('Pinned release seal SHA256 mismatch')
    release=json.loads(release_seal_path.read_text(encoding='utf-8'))
    if release.get('schema')!='CZ_CANONICAL_POSTHOC_RELEASE_SEAL_V1':
        raise ProtocolError('Unknown release seal schema')
    if file_sha(registration_path)!=release['registration_sha256']:
        raise ProtocolError('Registration bytes changed after sealing')
    registration=json.loads(registration_path.read_text(encoding='utf-8'))
    actual_sources={path.relative_to(module).as_posix() for path in module.rglob('*.py')
                    if '__pycache__' not in path.parts}
    if actual_sources!=set(release['canonical_source_sha256_lf']):
        raise ProtocolError('Runtime canonical source inventory changed')
    for relative,expected in release['canonical_source_sha256_lf'].items():
        if source_sha(_inside(module,relative))!=expected:
            raise ProtocolError('Runtime canonical source changed: '+relative)
    if set(release['canonical_source_sha256_lf'])!=set(registration['canonical_source_sha256_lf']):
        raise ProtocolError('Canonical source inventory changed')
    if release['canonical_source_sha256_lf']!=registration['canonical_source_sha256_lf']:
        raise ProtocolError('Canonical source hashes differ from registration')
    for relative,expected in release['authority_bindings_sha256_lf'].items():
        if source_sha(_inside(authority,relative))!=expected:
            raise ProtocolError('Runtime authority source/config changed: '+relative)
    if release['authority_bindings_sha256_lf']!=registration['authority_bindings_sha256_lf']:
        raise ProtocolError('Authority bindings differ from registration')
    for relative,expected in release['canonical_protocol_sha256'].items():
        if file_sha(_inside(module,relative))!=expected:
            raise ProtocolError('Runtime canonical protocol changed: '+relative)
    if release['canonical_protocol_sha256']!=registration['canonical_protocol_sha256']:
        raise ProtocolError('Canonical protocol inventory differs from registration')
    if registration['scientific_provenance']['revision_status']!='POST_HOC_PROTOCOL_REVISION':
        raise ProtocolError('Historical test exposure must be explicit')
    if not registration['scientific_provenance']['historical_target_test_used_for_protocol_design']:
        raise ProtocolError('Cannot relabel the previously viewed test as virgin')
    return registration,dict(release_sha256=expected_release_sha,
        registration_sha256=release['registration_sha256'],policy_id=digest(registration),
        canonical_sources_verified=len(release['canonical_source_sha256_lf']),
        authority_bindings_verified=len(release['authority_bindings_sha256_lf']))


def verify_c1(shared,binding):
    """Read hashes/metadata only; label projection remains a separate operation."""
    shared=Path(shared)
    manifest_path=shared/'C1_MAXIMAL_ANCHOR_MANIFEST.json'
    if file_sha(manifest_path)!=binding['manifest_sha256']:
        raise ProtocolError('C1 manifest changed')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['contract_id']!=binding['anchor_contract'] or manifest['train_support_hash']!=binding['train_support_hash']:
        raise ProtocolError('C1 anchor contract/support changed')
    files={binding['sample_relative']:manifest['train_metadata_parquet_sha256'],
        'JOINT_LIFT_PCA_CONTRACT.json':manifest['pca_sha256'],
        'base_data/cz_czochralski/train.parquet':manifest['source_base_sha256']}
    for relative,expected in files.items():
        # The frozen base-data symlink may intentionally refer to authority
        # storage; its bytes, rather than a mutable path alias, are sealed.
        if file_sha(shared/relative)!=expected:
            raise ProtocolError('C1 metadata/base/PCA changed: '+relative)
    if 'input_bytes_sha256' in binding:
        if str(Path(manifest['source_shared']).resolve())!=binding['inference_source_root']:
            raise ProtocolError('Inference input source identity changed')
        for relative,sha in binding['input_bytes_sha256'].items():
            if sha is None:
                if _inside(binding['inference_source_root'],relative).exists():
                    raise ProtocolError('Frozen absent inference input appeared: '+relative)
                continue
            if file_sha(_inside(binding['inference_source_root'],relative))!=sha:
                raise ProtocolError('Frozen inference input bytes changed: '+relative)
    return manifest
