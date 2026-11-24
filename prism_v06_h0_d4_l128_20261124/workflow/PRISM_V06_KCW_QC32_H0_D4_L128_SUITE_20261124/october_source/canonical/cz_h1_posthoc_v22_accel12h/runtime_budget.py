"""Persistent wall-time ledger; resume never grants a new formal budget."""
import json,time,os,signal,math
from pathlib import Path
from .invariants import dump,ProtocolError

class BudgetStop(RuntimeError):pass

class RuntimeBudget:
    def __init__(self,root,contract,*,clock=time.monotonic):
        self.path=Path(root)/'COMPUTE_WALL_TIME_LEDGER.json';self.contract=contract;self.clock=clock
        self.started=clock();self.active=True
        if self.path.exists():
            self.body=json.loads(self.path.read_text(encoding='utf-8'))
            if self.body['contract']!=contract:raise ProtocolError('Resume runtime budget changed')
            # Recover elapsed time of a killed segment conservatively using UTC.
            if self.body.get('active_since_utc') is not None:
                self.body['cumulative_seconds']+=max(0,time.time()-self.body['active_since_utc'])
        else:self.body=dict(contract=contract,cumulative_seconds=0.,segments=0)
        self.base=self.body['cumulative_seconds'];self.body['segments']+=1
        self.checkpoint()
    def elapsed(self):return self.base+max(0,self.clock()-self.started)
    def checkpoint(self,*,closed=False):
        self.body.update(cumulative_seconds=self.elapsed(),active_since_utc=None if closed else time.time())
        dump(self.path,self.body)
    def before_batch(self):
        elapsed=self.elapsed()
        if elapsed>=self.contract['hard_stop_seconds']:raise BudgetStop('HARD_STOP_BEFORE_12_HOURS')
        if elapsed>=self.contract['soft_stop_seconds']:raise BudgetStop('SOFT_STOP_NO_NEW_CANDIDATE_BATCH')
        self.checkpoint()
    def close(self):self.checkpoint(closed=True);self.active=False

def launch_gate(projection,protocol):
    override=protocol.get('launch_gate_user_override')
    if override is not None:
        expected=dict(schema='CZ_H1_EXPLICIT_USER_LAUNCH_OVERRIDE_V1',
            authorization_source='HUMAN_USER',formal_development_training=True,
            source_validation_reporting=False,target_test_inference=False,
            waive_pending_benchmark_completion=True,waive_prelaunch_runtime_forecast=True,
            scientific_rules_unchanged=True,runtime_stop_limits_unchanged=True)
        if (not isinstance(override,dict) or any(type(override.get(k)) is not type(v) or override.get(k)!=v for k,v in expected.items())
                or not isinstance(override.get('user_instruction'),str) or not override['user_instruction'].strip()):
            raise ProtocolError('Invalid explicit user launch override')
        return
    # A malformed or non-finite estimate must never open the formal gate.
    factor=projection.get('safety_factor')
    seconds=projection.get('projected_with_safety_factor')
    if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
           for value in (factor,seconds)):
        raise ProtocolError('TIME_BUDGET_GATE_FAILED: invalid runtime projection')
    if factor<1.25:raise ProtocolError('Runtime safety factor below frozen minimum')
    if seconds<0 or seconds>protocol['runtime_budget']['launch_gate_seconds']:
        raise ProtocolError('TIME_BUDGET_GATE_FAILED')

def watchdog(pid,root,contract,*,poll_seconds=1):
    """Separate supervisor terminates only a pinned own-run process at 11.5h."""
    root=Path(root);identity=Path(f'/proc/{pid}/stat').read_text().split()[21]
    while Path(f'/proc/{pid}/stat').exists():
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        if stat[21]!=identity or stat[2]=='Z':return
        ledger=json.loads((root/'COMPUTE_WALL_TIME_LEDGER.json').read_text(encoding='utf-8'))
        elapsed=ledger['cumulative_seconds']+max(0,time.time()-(ledger.get('active_since_utc') or time.time()))
        if elapsed>=contract['hard_stop_seconds']:
            os.kill(pid,signal.SIGINT)
            time.sleep(5)
            if Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split()[21]==identity:
                os.kill(pid,signal.SIGTERM)
            return
        if elapsed>=contract['soft_stop_seconds']:(root/'SOFT_STOP_REQUESTED').touch()
        time.sleep(poll_seconds)
