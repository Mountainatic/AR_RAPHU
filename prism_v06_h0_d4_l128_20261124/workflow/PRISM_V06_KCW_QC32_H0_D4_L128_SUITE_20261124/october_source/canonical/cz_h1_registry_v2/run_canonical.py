"""No-training entry point to the actual versioned canonical H1 engine.

There is deliberately no automatic training/old-test controller. Future H1
execution must use open_case and this same CanonicalPipeline, after its own
experiment selection/checkpoint/test barriers are registered.
"""
from pathlib import Path
import argparse
import json
import os
import sys


def open_case(root,shared,registration,*,no_training=True):
    from canonical_engine import CanonicalPipeline
    from prism_benchmark.cz_l256_nowcast import view
    from maximal_anchor import load
    samples,manifest=load(shared,for_fit=not no_training)
    if registration['training_support_hashes']['H1'][Path(shared).name]!=manifest['train_support_hash']:
        raise RuntimeError('Runner registration points at another C1 anchor universe')
    pipeline=CanonicalPipeline(root,shared,view('dynamic',h_steps=1,task_id='CZ_DIAM_RAW2S_CURRENT_L256_H1'),
                               'H1',registration,metadata_only=no_training)
    return pipeline,samples,manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight'])
    parser.add_argument('--authority-project',type=Path,required=True)
    parser.add_argument('--authority-shared-h1',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--junit',type=Path,required=True)
    args=parser.parse_args()
    os.environ['CZ_AUTHORITY_ROOT']=str(args.authority_project)
    from preflight import run
    run(args.authority_project,args.authority_shared_h1,args.output_root,args.junit)
