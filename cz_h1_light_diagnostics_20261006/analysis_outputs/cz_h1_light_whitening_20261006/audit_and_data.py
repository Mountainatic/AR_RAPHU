"""Read-only raw-data audit and reuse of frozen H1 data helper source."""
from pathlib import Path
import ast, hashlib, json, math
from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RAW = PROJECT / '实验数据1-张.xlsx'
SOURCE = PROJECT / 'analysis_outputs/PRISM_R2_H1_WHITENING_CORE_IMPLEMENTATION_20261002/authority_source_snapshot/src/prism_benchmark/cz_l256_nowcast.py'
HELPERS = ['RawSegment','_segment_ranges','_raw_segments','_split_source_origins','_fit_joint_lift','_apply_joint_lift']
HISTORY_STEPS = 256
H_STEPS = 1
BREAKPOINTS = {'Rod_1': (334,688,734), 'Rod_2': (745,)}
RAW_TO_CANONICAL = {'主加热功率':'main_heater_power','晶升速度':'crystal_lift','埚升速度':'crucible_lift','晶转速度':'crystal_rotation','埚转速度':'crucible_rotation','晶体直径':'crystal_diameter'}
TARGET_COLUMN = 'crystal_diameter'
_effective_h_steps = lambda h: 1 if h == 0 else h
_hash_text = lambda s: hashlib.sha256(s.encode()).hexdigest()
_tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
for _node in _tree.body:
    if isinstance(_node,(ast.FunctionDef,ast.ClassDef)) and _node.name in HELPERS:
        exec(compile(ast.Module(body=[_node],type_ignores=[]),str(SOURCE),'exec'),globals())

EXTRA = {'加热元件温度':'heater_temperature','氩气流量设定':'argon_flow_setpoint','晶体长度':'crystal_length','炉压':'furnace_pressure'}
COMMON = ['main_heater_power','joint_lift','crystal_rotation','crucible_rotation']
ALL = COMMON + ['crystal_lift','crucible_lift','heater_temperature','argon_flow_setpoint','crystal_length','furnace_pressure']

def write_json(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def load():
    sheets = pd.read_excel(RAW,sheet_name=None)
    rods = {f'Rod_{i+1}': sheets[f'Sheet{i+1}'].dropna(how='all').reset_index(drop=True) for i in range(2)}
    segments = {}
    for rod,raw in rods.items():
        segs = _raw_segments(rod,raw)
        for seg in segs:
            for raw_name,name in EXTRA.items():
                if raw_name in raw:
                    seg.frame[name] = raw.iloc[seg.raw_start:seg.raw_stop][raw_name].to_numpy(float)
        segments[rod] = segs
    return rods,segments

def audit():
    rods,segs = load()
    rows = []
    fields = sorted(set.union(*(set(r.columns) for r in rods.values())))
    categories = {'主加热功率':'control','晶升速度':'actuator','埚升速度':'actuator','晶转速度':'actuator','埚转速度':'actuator','加热元件温度':'measurement_context','氩气流量设定':'control_setpoint','晶体长度':'operating_state','炉压':'operating_state','晶体直径':'target_measurement'}
    for rod,raw in rods.items():
        for field in fields:
            exists = field in raw
            common = all(field in f for f in rods.values())
            allowed = exists and field != '晶体直径'
            col = raw[field] if exists else None
            rows.append(dict(rod=rod,field_name=field,canonical_name={**RAW_TO_CANONICAL,**EXTRA}[field],dtype=str(col.dtype) if exists else 'ABSENT',missing_ratio=float(col.isna().mean()) if exists else None,variance=float(col.var(ddof=0)) if exists else None,unique_count=int(col.nunique()) if exists else 0,Rod1_exists=field in rods['Rod_1'],Rod2_exists=field in rods['Rod_2'],common=common,Rod1_only=field in rods['Rod_1'] and not common,Rod2_only=field in rods['Rod_2'] and not common,category=categories[field],available_at_prediction='past rows <= t-1 only; record_time assumption' if exists else 'ABSENT',allowed_predictor=allowed,exclusion_reason='' if allowed else ('target/diameter excluded from all control-only arms' if exists else 'absent'),availability_evidence='recorded per-row process channel; workbook has no independent latency metadata' if allowed else ''))
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT/'VARIABLE_SCHEMA_AUDIT.csv',index=False,encoding='utf-8-sig')
    text = '# CZ H1 变量审计\n\n训练前生成。Rod1 20103 行、10 个字段；Rod2 20627 行、7 个字段。没有缺失值。全部数值采用原始单位。\n\n'
    text += 'Rod1-only：氩气流量设定、晶体长度、炉压。温度是两棒公共观测变量，但不在 Common4 中。AllLegal 固定使用四个正式输入，加两种原始升速、温度及三个 Rod1-only 字段，共 10 个通道。晶体直径不作为 predictor。\n\n'
    text += '合法性条件：沿用项目 record_time 约定，仅用 t-1 及更早记录；原工作簿没有独立传感器延迟或字段生成元数据。晶体长度按已记录工艺状态处理，没有利用直径构造它；如该字段实际为离线回填，需另一次预注册实验排除，不能根据本轮结果删变量。\n\n'
    text += 'joint_lift：原 cz_l256_nowcast.py 的 _fit_joint_lift / _apply_joint_lift 原样复用（AST 仅提取数据函数）；晶升/埚升训练历史行去重并标准化，取 PC1，晶升 loading 为非负。每个 fit 单独拟合 PCA，HISTORY_STEPS 设置为该候选 L；不读取未来/validation/test 行拟合。\n\n'
    text += '|rod|field|dtype|missing|variance|unique|category|legal|\n|---|---|---|---:|---:|---:|---|---|\n'
    for r in rows:
        if r['dtype'] != 'ABSENT': text += f"|{r['rod']}|{r['field_name']}|{r['dtype']}|{r['missing_ratio']:.3g}|{r['variance']:.6g}|{r['unique_count']}|{r['category']}|{r['allowed_predictor']}|\n"
    (ROOT/'VARIABLE_SCHEMA_AUDIT.md').write_text(text,encoding='utf-8')
    write_json('ROD1_ALL_LEGAL_VARIABLES.json',dict(status='FROZEN_BEFORE_TRAINING',variables=ALL,raw_fields=[c for c in rods['Rod_1'] if c != '晶体直径'],availability_policy='record_time; lag >=1; crystal_length assumed contemporaneous recorded state',raw_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()))
    members = []
    for rod,ss in segs.items():
        train,val = _split_source_origins(ss,h_steps=1)
        for split,groups in [('train',train),('source_validation',val)]:
            for entity,origins in groups.items():
                for origin in sorted(origins): members.append(dict(rod=rod,entity=entity,origin=origin,original_split=split))
        direction = f'{rod}_to_Rod_{2 if rod=="Rod_1" else 1}'
        frozen = pd.read_parquet(PROJECT/f'analysis_outputs/cz_h1_a_history_20261001/artifacts/models/{direction}/upstream/validation_train_only/fit_support.parquet')
        assert set(zip(frozen.entity_id,frozen.origin)) == {(e,o) for e,os in train.items() for o in os}, 'frozen train membership mismatch'
        prior_val = pd.read_parquet(PROJECT/f'analysis_outputs/cz_h1_a_history_20261001/artifacts/results/{direction}/validation/common_support.parquet')
        assert set(zip(prior_val.entity_id,prior_val.origin)) <= {(e,o) for e,os in val.items() for o in os}
    membership = pd.DataFrame(members)
    membership.to_parquet(ROOT/'ORIGINAL_SOURCE_MEMBERSHIP.parquet',index=False)
    print(membership.groupby(['rod','original_split']).origin.agg(['min','max','count']).to_string(),flush=True)
    return membership

if __name__ == '__main__': audit()
