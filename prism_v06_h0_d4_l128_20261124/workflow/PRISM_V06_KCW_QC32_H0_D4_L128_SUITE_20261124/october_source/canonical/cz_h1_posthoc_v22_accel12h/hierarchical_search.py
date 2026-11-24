"""Pre-target registered screen validation and replay worker identity transport."""
import json
from pathlib import Path
from .invariants import digest,file_sha,ProtocolError

def freeze_tuple(v):
    return tuple(freeze_tuple(x) for x in v) if isinstance(v,list) else v

def registered_screen(pipe):
    module=Path(__file__).parent
    contract=json.loads((module/'ACCEL_K_SEARCH_PROTOCOL.json').read_text(encoding='utf-8'))
    if file_sha(module/'K_MODEL_REGISTRY.json')!=contract['registry_sha256']:
        raise ProtocolError('Temporal screen registry drift')
    if not contract['frozen_before_new_revision_target_access']:
        raise ProtocolError('Temporal screen was not frozen before target access')
    specs=pipe.K_specs
    first=next(s for s in specs if s[2]=='RANK_1_URYSOHN')
    if digest(first)!=contract['template_recipe_id']:
        raise ProtocolError('Screen template is not first registered rank-one recipe')
    coords=list(dict.fromkeys(s[:2] for s in specs))
    screen=[next(s for s in specs if s[:2]==c and s[2:]==first[2:]) for c in coords]
    if len(screen)!=69 or [digest(s) for s in screen]!=contract['screen_recipe_ids']:
        raise ProtocolError('All 69 exact screen recipes must match the target-free freeze')
    return screen

class IdentityBinding:
    def __init__(self,value):self.value=value
    def identity(self):return self.value
